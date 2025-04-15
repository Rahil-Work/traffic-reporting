# project/core/video_processor.py
import cv2
import numpy as np
import torch
import time
import os
import threading
import queue
import gc
from datetime import datetime, timedelta
import concurrent.futures # <--- IMPORTED
from concurrent.futures import ThreadPoolExecutor

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, DEBUG_MODE, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS,
    MODEL_INPUT_SIZE # <-- Use configured input size
)

# Modules
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
from core.line_detection import Line, draw_lines_on_image # Import draw helper
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.vehicle_tracker import VehicleTracker
from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance


class VideoProcessor:
    def __init__(self):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE
        self.max_workers = THREAD_COUNT

        # Gradio state
        self.gradio_lines = {}
        self.current_gradio_direction = None
        self.gradio_drawing = False
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_lines_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.vehicle_tracker = None
        self.zone_tracker = None
        self.perf = None
        self.detection_lines = None # Populated based on mode
        self.model_input_size_config = MODEL_INPUT_SIZE # Store configured size

        # --- Define a lock specifically for tracker state modifications ---
        self.tracker_lock = threading.Lock()

        # Threading & Queues
        self.frame_read_queue = None
        self.video_write_queue = None
        self.stop_event = None
        self.reader_thread = None
        self.writer_thread = None
        self.last_cleanup_time = 0

        # CUDA Streams
        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0

        # Load hardcoded lines immediately if mode requires
        if self.line_mode == 'hardcoded':
            try:
                # Check if LINE_POINTS seems valid before creating objects
                if not LINE_POINTS or not all(isinstance(p, list) and len(p)==2 and isinstance(p[0], tuple) and isinstance(p[1], tuple) for p in LINE_POINTS.values()):
                     raise ValueError("LINE_POINTS invalid or incomplete in config.py")
                self.detection_lines = {
                    direction: Line(points, direction)
                    for direction, points in LINE_POINTS.items()
                }
                print("Init: Hardcoded lines loaded.")
            except Exception as e:
                 print(f"FATAL: Failed to initialize hardcoded lines: {e}")
                 raise SystemExit(f"Error initializing VideoProcessor: {e}")

    # --- Gradio Related Methods ---
    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive':
             status_msg = "Line mode is 'hardcoded'. Drawing disabled."
             frame = None; # Try loading frame for display
             try:
                 cap = cv2.VideoCapture(video_path)
                 if cap.isOpened(): ret, frame = cap.read(); cap.release()
                 if ret:
                     frame = cv2.resize(frame, self.frame_shape)
                     # Draw hardcoded lines for reference if available
                     lines_to_draw = self.detection_lines if self.detection_lines else {}
                     frame = draw_lines_on_image(frame, lines_to_draw)
                     return frame, status_msg
             except Exception as e: debug_print(f"Gradio upload frame load failed: {e}")
             return None, status_msg

        self.reset_gradio_state()
        if video_path is None: return None, "Please upload a video."
        try:
            cap = cv2.VideoCapture(video_path); ret, frame = cap.read(); cap.release()
            if ret:
                frame = cv2.resize(frame, self.frame_shape); self.original_frame_for_gradio = frame.copy(); self.last_drawn_lines_frame = frame.copy()
                return frame, "Video loaded. Select direction and click image to draw."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"

    def set_gradio_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_lines_frame, "Drawing disabled."
        self.current_gradio_direction = direction.lower(); self.gradio_drawing = False; self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click to start drawing line."
        frame_to_show = self.last_drawn_lines_frame if self.last_drawn_lines_frame is not None else self.original_frame_for_gradio
        return frame_to_show, status

    def handle_gradio_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_lines_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_lines_frame, "Select direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1])
        frame_copy = draw_lines_on_image(self.original_frame_for_gradio.copy(), self.gradio_lines)
        if not self.gradio_drawing:
            self.gradio_temp_points = [point]; self.gradio_drawing = True
            cv2.circle(frame_copy, point, 5, (0, 255, 255), -1); status = f"Point 1 for {self.current_gradio_direction.capitalize()} at {point}. Click second point."
        else:
            self.gradio_temp_points.append(point)
            try:
                self.gradio_lines[self.current_gradio_direction] = Line(self.gradio_temp_points, self.current_gradio_direction)
                self.gradio_drawing = False; frame_copy = draw_lines_on_image(self.original_frame_for_gradio.copy(), self.gradio_lines)
                status = f"Line for {self.current_gradio_direction.capitalize()} drawn."
            except ValueError as e: status = f"Error: {e}"; self.gradio_drawing = False
        self.last_drawn_lines_frame = frame_copy
        return self.last_drawn_lines_frame, status

    def reset_gradio_state(self):
        self.gradio_lines = {}; self.current_gradio_direction = None; self.gradio_drawing = False
        self.gradio_temp_points = []; self.original_frame_for_gradio = None; self.last_drawn_lines_frame = None

    # --- Video Reading/Writing Threads ---
    def _frame_reader_task(self, video_path, original_fps):
        """Reads frames from video file and puts them in the queue."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print("ERROR: Reader failed to open video."); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps)); frame_count = 0; frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read();
                if not ret: break
                if frame_count % frame_interval == 0:
                    frame_time_seconds = frame_count / original_fps
                    try:
                        while not self.stop_event.is_set():
                            try: self.frame_read_queue.put((frame, frame_count, frame_time_seconds), timeout=0.1); frames_yielded += 1; break
                            except queue.Full: continue
                    except Exception as e: debug_print(f"Reader queue error: {e}"); break
                frame_count += 1
            debug_print(f"Reader finished. Read {frame_count}, yielded ~{frames_yielded}.")
        except Exception as e: print(f"ERROR in reader thread: {e}"); import traceback; traceback.print_exc()
        finally:
            if cap: cap.release()
            if self.frame_read_queue: self.frame_read_queue.put(None)

    def _frame_writer_task(self, output_path):
        """Writes processed frames from queue to the output video file."""
        out = None; writer_initialized = False; frames_written = 0; actual_output_path = output_path
        try:
             fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.frame_shape)
             if not out.isOpened():
                 print(f"Warn: mp4v failed. Trying XVID/AVI."); actual_output_path = output_path.replace(".mp4", ".avi")
                 fourcc = cv2.VideoWriter_fourcc(*'XVID'); out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
             if not out.isOpened(): print(f"ERROR: Writer failed for: {actual_output_path}"); return
             writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")
             while True:
                try: frame_data = self.video_write_queue.get(timeout=1.0) # Longer timeout
                except queue.Empty:
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue
                if frame_data is None: break
                frame = frame_data
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                     if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                     if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                     if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]: frame = cv2.resize(frame, self.frame_shape)
                     out.write(frame); frames_written += 1
                else: debug_print(f"Writer received invalid frame: {type(frame_data)}")
                # Ensure task_done is called even for invalid frames
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e: print(f"ERROR in writer thread: {e}"); import traceback; traceback.print_exc()
        finally:
            if out and writer_initialized: out.release(); debug_print(f"Writer released. Wrote {frames_written} to {actual_output_path}")
            else: debug_print("Writer finished without writing/init.")


    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frame_list): # Uses MODEL_INPUT_SIZE
        tensors = []; target_size = self.model_input_size_config
        for frame in frame_list:
            img=cv2.resize(frame,(target_size,target_size)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=img.transpose(2,0,1); img=np.ascontiguousarray(img); tensors.append(img)
        batch_tensor=torch.from_numpy(np.stack(tensors)).to(self.device)
        dtype=torch.float16 if (MIXED_PRECISION and self.device=='cuda') else torch.float32
        batch_tensor = batch_tensor.to(dtype)/255.0; return batch_tensor

    def _process_inference_results(self, results, frames_batch, frame_numbers, frame_times, start_datetime): # Uses Parallelism & Miss Counting
        batch_detections_list = [[] for _ in range(len(results))]
        processed_frames_list = [None] * len(results)
        tracker_lock = self.tracker_lock # Use instance lock

        def process_single_frame(idx):
            result = results[idx]; original_frame = frames_batch[idx]; frame_number = frame_numbers[idx]
            frame_time_sec = frame_times[idx]; current_timestamp = start_datetime + timedelta(seconds=frame_time_sec)
            output_frame = cv2.resize(original_frame.copy(), self.frame_shape); frame_local_detections = []
            if not self.vehicle_tracker or not self.zone_tracker: print(f"ERR: Trackers missing frame {idx}"); return [], output_frame

            detections_this_frame = []; ids_processed_this_frame_map = {}
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0]); v_type_name = result.names.get(cls_id, f"CLS_{cls_id}")
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    model_h,model_w = result.orig_shape if hasattr(result,'orig_shape') else (self.model_input_size_config, self.model_input_size_config)
                    out_h,out_w = self.frame_shape[1], self.frame_shape[0]
                    x1o=max(0,int(x1*out_w/model_w)); y1o=max(0,int(y1*out_h/model_h)); x2o=min(out_w-1,int(x2*out_w/model_w)); y2o=min(out_h-1,int(y2*out_h/model_h))
                    center_pt = ((x1o+x2o)//2, (y1o+y2o)//2)
                    detections_this_frame.append({'box_coords': (x1o,y1o,x2o,y2o), 'center': center_pt, 'type': v_type_name})

            ids_seen_in_lock = set(); ids_to_remove_this_frame = set()
            with tracker_lock: # --- Start Lock ---
                if self.perf: self.perf.start_timer('tracking_update')
                for det_info in detections_this_frame:
                    center_point, v_type_name = det_info['center'], det_info['type']
                    matched_id = self.vehicle_tracker.find_best_match(center_point, frame_time_sec)
                    v_id, c_type = self.vehicle_tracker.update_track(matched_id if matched_id else self.vehicle_tracker._get_next_id(), center_point, frame_time_sec, v_type_name)
                    ids_processed_this_frame_map[v_id] = {'center':center_point, 'box':det_info['box_coords'], 'type':c_type, 'event':None, 'status':'detected'}
                    ids_seen_in_lock.add(v_id)

                for v_id in list(ids_processed_this_frame_map.keys()):
                    center_point, c_type = ids_processed_this_frame_map[v_id]['center'], ids_processed_this_frame_map[v_id]['type']
                    prev_pos=None
                    if v_id in self.vehicle_tracker.tracking_history and len(self.vehicle_tracker.tracking_history[v_id])>1: prev_pos=self.vehicle_tracker.tracking_history[v_id][-2]
                    if prev_pos and self.zone_tracker:
                        if self.perf: self.perf.start_timer('zone_checking')
                        event_type, event_dir = self.zone_tracker.check_zone_transition(prev_pos, center_point, v_id, frame_time_sec)
                        if self.perf: self.perf.end_timer('zone_checking')
                        if event_type=="ENTRY":
                           if self.vehicle_tracker.register_entry(v_id,event_dir,current_timestamp,center_point,c_type):
                              if self.perf:self.perf.record_vehicle_entry(); d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{c_type} ENTERED FROM {event_dir.upper()}",'frame_number':frame_number,'vehicle_id':v_id,'status':'entry'})
                              ids_processed_this_frame_map[v_id]['event']='ENTRY'
                        elif event_type=="EXIT":
                           entry_dir_trk=self.vehicle_tracker.active_vehicles.get(v_id,{}).get('entry_direction')
                           if entry_dir_trk and is_valid_movement(entry_dir_trk,event_dir):
                              success,t_in_int=self.vehicle_tracker.register_exit(v_id,event_dir,current_timestamp,center_point)
                              if success:
                                 if self.perf:self.perf.record_vehicle_exit('exited',t_in_int); d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{c_type} EXITED TO {event_dir.upper()} (FROM {entry_dir_trk.upper()})",'frame_number':frame_number,'vehicle_id':v_id,'status':'exit','time_in_intersection':t_in_int})
                                 ids_processed_this_frame_map[v_id]['event']='EXIT'; ids_to_remove_this_frame.add(v_id)

                timed_out_ids=self.vehicle_tracker.check_timeouts(current_timestamp)
                for v_id_to in timed_out_ids:
                    ids_to_remove_this_frame.add(v_id_to);
                    if self.perf:self.perf.record_vehicle_exit('timed_out')
                    p_data=next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'),None)
                    if p_data: d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{p_data['type']} TIMED OUT (FROM {p_data['entry_direction'].upper()})",'frame_number':frame_number,'vehicle_id':v_id_to,'status':'timeout','time_in_intersection':p_data['time_in_intersection']})
                    if v_id_to in ids_processed_this_frame_map: ids_processed_this_frame_map[v_id_to]['status']='timed_out'

                self.vehicle_tracker.increment_misses(ids_seen_in_lock) # Call miss counter

                for v_id_rem in ids_to_remove_this_frame: self.vehicle_tracker.remove_vehicle_data(v_id_rem); self.zone_tracker.remove_vehicle_data(v_id_rem)
                if self.perf: self.perf.end_timer('tracking_update')
            # --- End Lock ---

            # --- Drawing Pass ---
            if self.perf: self.perf.start_timer('drawing')
            if self.zone_tracker: output_frame = visual_overlay.draw_zones(output_frame, self.zone_tracker)
            active_vehicle_snapshot = self.vehicle_tracker.active_vehicles.copy()
            for v_id_draw, data in ids_processed_this_frame_map.items():
                center_pt_draw, box_draw, c_type_draw = data['center'], data['box'], data['type']
                current_draw_status = data.get('status', 'detected'); display_status = "detected"; entry_dir_draw = None; t_active_draw = None
                if v_id_draw in active_vehicle_snapshot:
                     v_state = active_vehicle_snapshot[v_id_draw]
                     if v_state.get('status') == 'active': display_status='active'; entry_dir_draw=v_state['entry_direction']; t_active_draw=(current_timestamp - v_state['entry_time']).total_seconds()
                if current_draw_status != 'timed_out': visual_overlay.draw_detection_box(output_frame, box_draw, v_id_draw, c_type_draw, status=display_status, entry_dir=entry_dir_draw, time_active=t_active_draw)
                trail = self.vehicle_tracker.get_tracking_trail(v_id_draw); visual_overlay.draw_tracking_trail(output_frame, trail, v_id_draw)
                event_marker = data.get('event');
                if event_marker: visual_overlay.draw_event_marker(output_frame, center_pt_draw, event_marker, v_id_draw[-4:])
            output_frame = visual_overlay.add_status_overlay(output_frame, frame_number, current_timestamp, self.vehicle_tracker)
            if self.perf: self.perf.end_timer('drawing')
            return frame_local_detections, output_frame
            # --- End process_single_frame ---

        # --- Execute in parallel ---
        if self.perf: self.perf.start_timer('detection_processing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
             futures = {executor.submit(process_single_frame, i): i for i in range(len(results))}
             for future in concurrent.futures.as_completed(futures):
                 idx = futures[future]
                 try: frame_detections, processed_frame = future.result(); batch_detections_list[idx] = frame_detections; processed_frames_list[idx] = processed_frame
                 except Exception as exc: print(f'Frame {idx} failed: {exc}'); import traceback; traceback.print_exc(); processed_frames_list[idx] = cv2.resize(frames_batch[idx].copy(), self.frame_shape)
        final_batch_detections = [det for frame_list in batch_detections_list for det in frame_list]
        final_processed_frames = processed_frames_list
        if self.perf: self.perf.end_timer('detection_processing')
        return final_batch_detections, final_processed_frames
    # --- End _process_inference_results ---

    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str):
        if not self.processing_lock.acquire(blocking=False):
            # Already processing, prevent concurrent runs
            print("Warning: Processing lock already acquired. Another process may be running.")
            return "Processing is already in progress. Please wait."

        # --- Reset stateful components for a new run ---
        self.vehicle_tracker = VehicleTracker()
        self.zone_tracker = None # Initialized after lines determined
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        # Increased queue sizes for potentially faster I/O overlap
        self.frame_read_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.video_write_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.last_cleanup_time = time.monotonic()
        self.detection_lines = None # Reset detection lines for this run
        output_path = None # Define output_path
        self.final_output_path = None # Store the path used by the writer

        try:
            # --- Determine and Validate Lines ---
            if self.line_mode == 'hardcoded':
                if not LINE_POINTS or not all(LINE_POINTS.values()):
                    raise ValueError("Hardcoded line mode selected, but LINE_POINTS not fully defined in config.py")
                self.detection_lines = {
                    direction: Line(points, direction)
                    for direction, points in LINE_POINTS.items()
                }
                print("Using hardcoded lines from config.py")
            elif self.line_mode == 'interactive':
                if not self.gradio_lines or not all(ln for ln in self.gradio_lines.values()):
                    missing = [d for d in ['north', 'south', 'east', 'west'] if d not in self.gradio_lines or not self.gradio_lines[d]]
                    raise ValueError(f"Interactive line mode selected, but lines not drawn for all directions. Missing: {missing}")
                self.detection_lines = self.gradio_lines # Use lines from Gradio state
                print("Using lines drawn via Gradio interface.")
            else:
                raise ValueError(f"Invalid LINE_MODE '{self.line_mode}' configured.")

            # --- Initialize ZoneTracker *after* lines are determined ---
            self.zone_tracker = ZoneTracker(self.detection_lines)
            if not self.zone_tracker.zones:
                 print("Warning: ZoneTracker failed to create zones from the provided lines.")
                 # Processing might continue but zone crossings won't work

            # --- Validation & Initialization (Video, Time, Model) ---
            if video_path is None: raise ValueError("Video file path is missing.")
            print(f"Starting video processing (Lines: {self.line_mode})...")

            # Load model (only if not already loaded - though current logic re-inits processor)
            if self.model is None:
                self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model from {MODEL_PATH}")

            try: # Parse time
                start_date = datetime.strptime(start_date_str, "%y%m%d")
                time_part, ms_part = (start_time_str[:6], start_time_str[6:]) if len(start_time_str) > 6 else (start_time_str, "0")
                base_time = datetime.strptime(time_part, "%H%M%S").time()
                microseconds = int(ms_part.ljust(3,'0')) * 1000 # Pad ms if needed
                start_time = base_time.replace(microsecond=microseconds)
                start_datetime = datetime.combine(start_date.date(), start_time)
            except ValueError as e: raise ValueError(f"Invalid date/time format: {e}. Use YYMMDD and HHMMSS[mmm].")

            # Get video properties
            cap_check = cv2.VideoCapture(video_path);
            if not cap_check.isOpened(): raise FileNotFoundError(f"Cannot open video file: {video_path}")
            original_fps = cap_check.get(cv2.CAP_PROP_FPS); total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT)); cap_check.release()
            if original_fps <= 0: raise ValueError(f"Invalid original video FPS detected ({original_fps}).")

            # Setup output
            os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(VIDEO_OUTPUT_DIR, f"output_{ts}.mp4"); self.final_output_path = output_path # Store intended path

            # --- Start Threads ---
            print("Starting reader and writer threads...");
            self.reader_thread = threading.Thread(target=self._frame_reader_task, args=(video_path, original_fps), daemon=True)
            self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path,), daemon=True)
            self.reader_thread.start(); self.writer_thread.start()

            # --- Main Processing Loop ---
            all_detections = []; frames_processed_count = 0
            frame_interval = max(1, round(original_fps / self.target_fps))
            total_target_frames = total_frames // frame_interval if frame_interval > 0 else total_frames
            if self.perf: self.perf.start_processing(total_target_frames)
            frames_batch, frame_numbers_batch, frame_times_batch = [], [], []
            last_progress_print_time = time.monotonic()

            while True: # Loop until reader queue signals end
                try:
                    # --- Get Frame ---
                    frame_data = self.frame_read_queue.get(timeout=60) # Wait up to 60s for a frame
                    if frame_data is None: # End signal from reader
                        self.frame_read_queue.task_done()
                        break
                    if self.perf: self.perf.start_timer('video_read')
                    frame, frame_number, frame_time_sec = frame_data
                    if self.perf: self.perf.end_timer('video_read')

                    frames_batch.append(frame)
                    frame_numbers_batch.append(frame_number)
                    frame_times_batch.append(frame_time_sec)

                    # --- Process Batch ---
                    if len(frames_batch) >= self.batch_size:
                        batch_start_mono = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch = self._preprocess_batch(frames_batch);
                        if self.perf: self.perf.end_timer('preprocessing')

                        # --- Inference ---
                        stream = None; current_stream_idx = 0 # Placeholder if not using streams
                        if self.device == 'cuda' and self.streams: stream = self.streams[self.current_stream_index%len(self.streams)]; current_stream_idx=self.current_stream_index; self.current_stream_index+=1
                        # Use context manager for stream handling
                        with torch.cuda.stream(stream) if stream else torch.no_grad():
                           with torch.amp.autocast(device_type=self.device, enabled=MIXED_PRECISION and self.device=='cuda'):
                              if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream)
                              results = self.model(input_batch, verbose=False) # Get results
                              if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                           if stream: stream.synchronize() # Sync ONLY if using streams
                           if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                        # --- Post-processing (Parallel) ---
                        batch_log, processed_frames = self._process_inference_results(results, frames_batch, frame_numbers_batch, frame_times_batch, start_datetime)
                        all_detections.extend(batch_log)
                        if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in results if hasattr(r,'boxes') and r.boxes))

                        # --- Write Output ---
                        if self.perf: self.perf.start_timer('video_write');
                        for pf in processed_frames: self.video_write_queue.put(pf) # Add to writer queue
                        if self.perf: self.perf.end_timer('video_write')

                        # --- Stats & Cleanup ---
                        batch_time = time.monotonic() - batch_start_mono; frames_processed_count += len(frames_batch)
                        if self.perf: self.perf.record_batch_processed(len(frames_batch), batch_time); self.perf.sample_system_metrics()
                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_f_time = frame_times_batch[-1] if frame_times_batch else 0
                            cleanup_tracking_data(self.vehicle_tracker, self.zone_tracker, last_f_time); cleanup_memory()
                            self.last_cleanup_time = current_mono_time
                            if self.perf: self.perf.end_timer('memory_cleanup')

                        # --- Progress Update ---
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0): # Throttle prints
                             prog = self.perf.get_progress(); print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active:{self.vehicle_tracker.get_active_vehicle_count()}|ETA:{prog['eta']}", end="")
                             last_progress_print_time = current_mono_time

                        # --- Clear Batch ---
                        frames_batch.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()
                        del input_batch, results; cleanup_memory() # Explicit cleanup

                    # Mark task as done in the reader queue
                    self.frame_read_queue.task_done()

                except queue.Empty:
                    # This shouldn't happen often with a long timeout unless the reader thread died
                    print("\nWarning: Frame reader queue timed out waiting for frame. Reader might have stopped.")
                    if not self.reader_thread.is_alive():
                         print("Reader thread is not alive. Stopping processing.")
                         break
                    else:
                         print("Reader thread still alive, continuing wait...")
                         continue # Keep waiting if reader is alive
                except Exception as e:
                     print(f"\nERROR during processing loop: {e}")
                     import traceback; traceback.print_exc()
                     break # Exit loop on error

            # --- Process Final Batch ---
            if frames_batch:
                print("\nProcessing final batch...")
                batch_start_mono = time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing'); input_batch=self._preprocess_batch(frames_batch); self.perf.end_timer('preprocessing')
                stream=None; idx=0;
                if self.device=='cuda' and self.streams: stream=self.streams[self.current_stream_index%len(self.streams)]; idx=self.current_stream_index; self.current_stream_index+=1
                with torch.cuda.stream(stream) if stream else torch.no_grad():
                   with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                      if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream); results=self.model(input_batch,verbose=False);
                      if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                   if stream: stream.synchronize()
                   if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                batch_log, processed_frames = self._process_inference_results(results, frames_batch, frame_numbers_batch, frame_times_batch, start_datetime)
                all_detections.extend(batch_log)
                if self.perf: self.perf.start_timer('video_write');
                for pf in processed_frames: self.video_write_queue.put(pf)
                if self.perf: self.perf.end_timer('video_write')
                batch_time=time.monotonic()-batch_start_mono; frames_processed_count+=len(frames_batch)
                if self.perf: self.perf.record_batch_processed(len(frames_batch),batch_time)
                del input_batch, results; cleanup_memory()

            print("\nVideo processing loop finished.")

            # --- Finalize & Report ---
            if self.vehicle_tracker and self.vehicle_tracker.active_vehicles: # Check if tracker exists
                 print(f"Forcing exit for {len(self.vehicle_tracker.active_vehicles)} remaining...")
                 last_ts = start_datetime + timedelta(seconds=frame_times_batch[-1]) if frame_times_batch else start_datetime
                 last_fnum = frame_numbers_batch[-1] if frame_numbers_batch else (total_frames -1)
                 force_logs = []
                 # Use list copy for safe iteration while removing
                 for v_id in list(self.vehicle_tracker.active_vehicles.keys()):
                     if self.vehicle_tracker.force_exit_vehicle(v_id, last_ts):
                         if self.perf: self.perf.record_vehicle_exit('forced_exit')
                         p_data = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id']==v_id and p['status']=='forced_exit'),None)
                         if p_data: date_str, time_str_ms = format_timestamp(last_ts); force_logs.append({'timestamp':time_str_ms, 'date':date_str, 'detection':f"{p_data['type']} FORCED EXIT (FROM {p_data['entry_direction'].upper()})", 'frame_number':last_fnum, 'vehicle_id':v_id, 'status':'forced_exit', 'time_in_intersection':p_data['time_in_intersection']})
                         # Ensure removal after force exit processing
                         self.vehicle_tracker.remove_vehicle_data(v_id)
                         if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id)
                 all_detections.extend(force_logs)

            # --- Signal threads and wait ---
            print("Signaling threads to stop...")
            if self.stop_event: self.stop_event.set()
            # Put None signals to ensure threads unblock if waiting on queue
            if self.frame_read_queue:
                try: self.frame_read_queue.put_nowait(None)
                except queue.Full: pass # Ignore if full, stop_event should handle it
            if self.video_write_queue:
                try: self.video_write_queue.put_nowait(None)
                except queue.Full: pass

            print("Waiting for threads to complete...")
            t_wait_start = time.monotonic()
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=5)
            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=20) # Give writer more time
            print(f"Threads joined in {time.monotonic()-t_wait_start:.2f}s")

            # --- Performance Summary & Reporting ---
            total_time_taken = 0.0; final_fps = 0.0
            completed_paths_count_valid = 0 # Count of ONLY valid exit paths

            if self.perf:
                 self.perf.end_processing()
                 self.perf.print_summary() # Print detailed summary to console
                 # Get specific values for Gradio output
                 total_time_taken = self.perf.total_time
                 if total_time_taken > 0:
                     final_fps = frames_processed_count / total_time_taken
            # Estimate if perf tracker disabled
            elif hasattr(self, 'start_time_proc'): # Check if basic timing was done
                 total_time_taken = time.time() - self.start_time_proc # Use basic timing if available
                 if total_time_taken > 0:
                      final_fps = frames_processed_count / total_time_taken

            # Get completed paths data from tracker
            final_completed_paths_list = []
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker:
                 final_completed_paths_list = self.vehicle_tracker.get_completed_paths()

                 # --- Calculate count of VALID paths ---
                 for path in final_completed_paths_list:
                     status = path.get('status')
                     entry_dir = path.get('entry_direction')
                     exit_dir = path.get('exit_direction')
                     is_valid_entry = entry_dir and entry_dir not in ['UNKNOWN', None]
                     is_valid_exit = exit_dir and exit_dir not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None]
                     if status == 'exited' and is_valid_entry and is_valid_exit:
                         completed_paths_count_valid += 1
            # Pass the aggregated detailed event list to the report function
            excel_path = create_excel_report(
                detection_events=all_detections,
                completed_paths_data=final_completed_paths_list, # Pass the full list from tracker
                start_datetime=start_datetime
            )
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)

            # --- Prepare result message WITH Stats ---
            result_msg = f"✅ Processing completed.\n"
            result_msg += f"Output video in '{VIDEO_OUTPUT_DIR}' (check console).\n"
            if excel_path: result_msg += f"Excel report: {excel_path}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS: result_msg += f"Perf charts: 'performance_charts/'\n"
            # Add parseable stats section
            result_msg += f"\n--- Summary Stats ---\n"
            result_msg += f"STAT_FRAMES_PROCESSED={frames_processed_count}\n"
            result_msg += f"STAT_TIME_SECONDS={total_time_taken:.2f}\n"
            result_msg += f"STAT_FPS={final_fps:.2f}\n"
            result_msg += f"STAT_COMPLETED_PATHS={completed_paths_count_valid}\n"
            # --- End Stats Section ---

            return result_msg

        except Exception as e:
            print(f"\nFATAL ERROR during video processing: {e}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set() # Signal threads to stop on error
            return f"❌ Error: {e}"
        finally:
            # --- Final Cleanup ---
            # Ensure threads are joined even if error occurred before join block
            try:
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
            except Exception as join_e: print(f"Error during final thread join: {join_e}")
            # Release resources
            self.model=None; self.vehicle_tracker=None; self.zone_tracker=None; self.perf=None; self.detection_lines=None
            self.frame_read_queue=None; self.video_write_queue=None; self.reader_thread=None; self.writer_thread=None; self.stop_event=None
            cleanup_memory();
            if self.processing_lock.locked():
                 try:
                     self.processing_lock.release()
                     print("Processing finished, lock released.")
                 except RuntimeError as release_err:
                      # This might happen if another thread released it unexpectedly, but unlikely
                      print(f"Warning: Error releasing processing lock: {release_err}")
            else:
                 print("Processing finished (lock was already released).")

# --- End of VideoProcessor class ---