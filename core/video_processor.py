# project/core/video_processor.py
import cv2
import numpy as np
import torch
import time
import os
import threading
import queue
import gc
import shutil # For directory cleanup
import math # For ceiling function
import subprocess # For ffmpeg muxing if needed by writer
from datetime import datetime, timedelta
from pathlib import Path # For paths
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, DEBUG_MODE, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION,
    # New Chunking/Encoding Config
    ENABLE_AUTO_CHUNKING, AUTO_CHUNK_THRESHOLD_MINUTES,
    AUTO_CHUNK_DURATION_MINUTES, CHUNK_TEMP_DIR, REPORT_TEMP_DIR,
    FFMPEG_PATH, ENCODER_CODEC, ENCODER_BITRATE, ENCODER_PRESET,
    RAW_STREAM_FILENAME, FINAL_VIDEO_EXTENSION, REPORT_OUTPUT_DIR, # Make sure REPORT_OUTPUT_DIR is imported
    # Tracker Type (though deferred)
    TRACKER_TYPE #, STRONGSORT_WEIGHTS_PATH # Add if/when using StrongSORT/BoxMOT
)

# Modules
from utils import (debug_print, format_timestamp, is_valid_movement,
                    cleanup_memory, get_video_properties, split_video_ffmpeg)
from performance import PerformanceTracker
from models.model_loader import load_model, get_device

# Import tracker based on config OR default to VehicleTracker
if TRACKER_TYPE == 'Custom':
    from tracking.vehicle_tracker import VehicleTracker

from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data

# Conditional GPU Accelerated Components
GPU_VIZ_ENABLED = False
NvidiaFrameReader = None
NvidiaFrameWriter = None
add_gpu_overlays = None

# Use GPU Overlays if Visualizing with Nvidia Writer
if ENABLE_VISUALIZATION:
    try:
        from visualization.gpu_overlay import add_gpu_overlays
        from .nvidia_reader import NvidiaFrameReader
        from .nvidia_writer import NvidiaFrameWriter
        GPU_VIZ_ENABLED = True
    except ImportError as e:
        print(f"Warning: Failed to import GPU Reader/Writer/Overlay components ({e}).")
        print("If ENABLE_VISUALIZATION is True, will attempt CPU fallback for visualization (slower).")
        # Fallback to CPU visualization method (original overlay)
        from visualization import overlay as visual_overlay
else:
    print("ENABLE_VISUALIZATION is False. No video output will be generated.")
    from visualization import overlay as visual_overlay # Needed for add_status_overlay even if video isn't written

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance

from reporting.excel_report import create_excel_report, consolidate_excel_reports

class VideoProcessor:
    def __init__(self):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE
        self.max_workers = THREAD_COUNT

        # Gradio Polygon State
        self.gradio_polygons = {} # Stores {direction: List[tuple(x,y)]}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.model_names = {} # Store model class names

        self.tracker_type_internal = TRACKER_TYPE
        self.tracker_lock = threading.Lock() # Might still be useful for custom tracker logic

        # Attributes managed per file/chunk run by _process_single_video_file
        self.tracker = None
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.frame_read_queue = None
        self.video_write_queue = None # For CPU VideoWriter (old) or NvidiaFrameWriter's input
        self.stop_event = None
        self.reader_thread = None
        self.writer_instance = None # Can be NvidiaFrameWriter or old CPU writer thread
        self.writer_thread = None # For CPU writer
        self.last_cleanup_time = 0
        self.final_output_path_current_file = None # Path for video output of the current file/chunk

        # CUDA Streams
        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0

        # Load model once
        try:
            print("VideoProcessor: Initializing Model...")
            self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError("Model loading failed during init.")
            # Store model names if available after loading
            self.model_names = getattr(self.model, 'names', {})
            if not self.model_names:
                print("VideoProcessor: Warning - Could not get class names from loaded model.")
            print("VideoProcessor: Model loaded successfully in __init__.")
        except Exception as e:
            print(f"FATAL: Error loading model during VideoProcessor init: {e}")
            self.model = None # Indicate model loading failure

    # --- Gradio Polygon Related Methods ---

    def _draw_polygons_for_gradio(self, frame_to_draw_on):
        if frame_to_draw_on is None: return None
        img_copy = frame_to_draw_on.copy()
        colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} # BGR
        default_color = (255, 255, 255); temp_color = (0, 255, 255)

        # Draw completed polygons
        for direction, points in self.gradio_polygons.items():
            if points and len(points) >= 3:
                pts_np = np.array(points, dtype=np.int32)
                color = colors.get(direction, default_color)
                cv2.polylines(img_copy, [pts_np], isClosed=True, color=color, thickness=2)
                try:
                    if len(pts_np) > 0:
                        M = cv2.moments(pts_np)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                            cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass # Ignore label drawing errors

        # Draw temporary points
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2: # Closing hint
                last_pt = self.gradio_temp_points[-1]; first_pt = self.gradio_temp_points[0]
                hint_color = tuple(c // 2 for c in current_temp_color)
                cv2.line(img_copy, last_pt, first_pt, hint_color, 1, cv2.LINE_AA)

        self.last_drawn_polygons_frame = img_copy
        return img_copy

    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive': return None, "Zone definition mode is 'hardcoded'. Drawing disabled."
        self.reset_gradio_polygons()
        if video_path is None: return None, "Please upload a video."
        try:
            cap=cv2.VideoCapture(video_path); ret,frame=cap.read(); cap.release()
            if ret:
                frame = cv2.resize(frame, self.frame_shape)
                self.original_frame_for_gradio = frame.copy()
                drawn_frame = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
                return drawn_frame, "Video loaded. Select direction and click image to define zone vertices."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"

    def set_gradio_polygon_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        direction_lower = direction.lower()
        if direction_lower in self.gradio_polygons:
             del self.gradio_polygons[direction_lower]
             print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower
        self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1])
        self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3:
            status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."
            frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
            return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy()
        status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else:
            removed_point = self.gradio_temp_points.pop()
            status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def reset_gradio_polygons(self):
        self.gradio_polygons = {}; self.current_gradio_direction = None
        self.gradio_temp_points = []; status = "All polygons cleared."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio) if self.original_frame_for_gradio is not None else None
        self.last_drawn_polygons_frame = frame_to_show
        return frame_to_show, status

    # --- Video Reading/Writing Threads ---
    def _frame_reader_task(self, video_path, original_fps):
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
                        # Add frame to queue for processing
                        self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                        frames_yielded += 1
                    except queue.Full:
                        debug_print("Reader queue full, waiting...")
                        time.sleep(0.1);
                        if self.stop_event.is_set(): break
                        try: # Retry put
                            self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                            frames_yielded += 1
                        except queue.Full: print("ERROR: Reader queue persistently full."); break
                    except Exception as e: debug_print(f"Reader queue error: {e}"); break
                frame_count += 1
            debug_print(f"Reader finished. Read {frame_count}, yielded ~{frames_yielded}.")
        except Exception as e: print(f"ERROR in reader thread: {e}")
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, timeout=1.0) # Signal end to reader queue
                except queue.Full: pass

    def _frame_writer_task(self, output_path):
        # This task will only run if ENABLE_VISUALIZATION is True
        out = None; writer_initialized = False; frames_written = 0
        actual_output_path = output_path
        try:
            # Attempt to initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI.");
                avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID');
                out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened():
                    actual_output_path = avi_path;
                    # Important: Update the class variable if the path changed
                    self.final_output_path = avi_path
                else:
                    print(f"ERROR: Writer failed to open for both MP4 and AVI.");
                    # Signal failure (e.g., by setting final_output_path to None or an error string)
                    self.final_output_path = "ERROR_WRITER_FAILED"
                    return # Exit thread if writer cannot be opened

            writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")

            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0) # Wait for frames
                except queue.Empty:
                    # Check if processing is done and queue is empty
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue # Continue waiting if not stopped

                if frame_data is None: break # Check for sentinel value

                frame = frame_data
                # Perform checks and write frame
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    # Ensure frame matches output shape before writing
                    if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]:
                        frame = cv2.resize(frame, self.frame_shape)
                    out.write(frame); frames_written += 1

                # Mark task as done only if queue exists
                if self.video_write_queue: self.video_write_queue.task_done()

        except Exception as e:
            print(f"ERROR in writer thread: {e}")
            # Signal failure
            if self.final_output_path != "ERROR_WRITER_FAILED": # Avoid overwriting specific error
                 self.final_output_path = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            # Release writer if it was initialized
            if out and writer_initialized:
                 out.release();
                 debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            # Signal writer queue end only if it exists (ensures main thread doesn't block indefinitely)
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass


    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frame_tensor_list):
        """ Prepares a batch from a list of frame tensors already on the GPU. """
        if not frame_tensor_list: return None
        processed_tensors = []
        target_h, target_w = self.model_input_size_config, self.model_input_size_config

        for frame_tensor in frame_tensor_list:
            if frame_tensor is None: continue
            if str(frame_tensor.device) != str(self.device):
                frame_tensor = frame_tensor.to(self.device)

            # Assuming frame_tensor is [C, H, W] from NvidiaReader
            _, current_h, current_w = frame_tensor.shape
            if current_h != target_h or current_w != target_w:
                frame_tensor = F.interpolate(
                    frame_tensor.unsqueeze(0), size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            processed_tensors.append(frame_tensor)

        if not processed_tensors: return None
        batch_tensor = torch.stack(processed_tensors)
        dtype = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        
        # Assuming NvidiaReader provides tensors in [0, 255] range. Normalize if needed.
        # If reader provides float [0,1], then this division is not needed.
        # For now, let's assume PyNvDecoder outputs uint8 surfaces, converted to tensors,
        # so they are effectively [0,255] range before this.
        batch_tensor = batch_tensor.to(dtype) / 255.0
        return batch_tensor

    def _process_inference_results(self, results_yolo, input_batch_tensor_for_viz, frame_numbers, frame_times, start_datetime_chunk):
        """
        Processes YOLO results for a batch, runs tracking, zone checks.
        Returns:
            - batch_structured_log_data (list of dicts): For Excel reporting.
            - batch_detections_for_gpu_viz (list of lists): Data for gpu_overlay.add_gpu_overlays.
                                                        Each inner list corresponds to a frame and contains dicts:
                                                        {'box': [x1,y1,x2,y2], 'id': 'v_xx', 'type': 'Car', 'status': 'active'}
        """
        batch_size_actual = len(results_yolo)
        batch_structured_log_data = [[] for _ in range(batch_size_actual)]
        batch_detections_for_gpu_viz = [[] for _ in range(batch_size_actual)]

        def process_single_frame_logic(idx):
            result = results_yolo[idx]
            frame_number_abs = frame_numbers[idx] # Absolute frame number in original video
            frame_time_sec_abs = frame_times[idx] # Time offset from original video start
            current_timestamp_abs = start_datetime_chunk + timedelta(seconds=frame_time_sec_abs) # Absolute timestamp

            frame_log_entries = []  # For this frame's Excel log data
            frame_viz_entries = []  # For this frame's GPU visualization data

            if not self.tracker or not self.zone_tracker:
                return [], []

            # 1. Extract Detections from YOLO results
            detections_for_tracker = [] # List of {'box_coords': (x1o,y1o,x2o,y2o), 'center':pt, 'type':name}
            # Assuming results are scaled to the input_batch_tensor size (model_input_size_config)
            tensor_h, tensor_w = input_batch_tensor_for_viz.shape[2], input_batch_tensor_for_viz.shape[3]

            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    v_type_name = self.model_names.get(cls_id, f"CLS_{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # Ensure coords are within tensor bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(tensor_w - 1, x2), min(tensor_h - 1, y2)
                    
                    center_pt = ((x1 + x2) // 2, (y1 + y2) // 2) # Center for matching
                    bottom_center_pt = ((x1 + x2) // 2, y2) # Bottom-center for zone check convention

                    detections_for_tracker.append({
                        'box_coords': (x1, y1, x2, y2),
                        'center': center_pt,
                        'bottom_center': bottom_center_pt, # For zone check logic
                        'type': v_type_name,
                        'confidence': float(box.conf[0])
                    })

            # --- Existing Custom Tracker Logic (adapted) ---
            ids_seen_this_frame_custom = set()
            processed_tracks_this_frame_map = {} # {v_id: {data for viz and zone check}}

            if self.tracker_type_internal == 'Custom': # Assuming self.tracker is VehicleTracker instance
                # Match existing tracks
                matched_ids_map = {} # det_idx -> v_id
                unmatched_detections_indices = []
                for det_idx, det_info in enumerate(detections_for_tracker):
                    # Use 'center' or 'bottom_center' for find_best_match based on its implementation
                    # Your VehicleTracker.find_best_match expects center_point and frame_time_sec
                    matched_id = self.tracker.find_best_match(det_info['center'], frame_time_sec_abs)
                    if matched_id: matched_ids_map[det_idx] = matched_id
                    else: unmatched_detections_indices.append(det_idx)

                # Update matched tracks
                for det_idx, v_id in matched_ids_map.items():
                    det_info = detections_for_tracker[det_idx]
                    v_id, consistent_type = self.tracker.update_track(
                        v_id, det_info['center'], frame_time_sec_abs, det_info['type']
                    )
                    processed_tracks_this_frame_map[v_id] = {
                        'box': det_info['box_coords'], 'type': consistent_type, 'id': v_id,
                        'center_for_zone': det_info['bottom_center'], # Use bottom-center for zone check
                        'event': None, 'status': 'detected'
                    }
                    ids_seen_this_frame_custom.add(v_id)

                # Handle new tracks
                for det_idx in unmatched_detections_indices:
                    det_info = detections_for_tracker[det_idx]
                    v_id = self.tracker._get_next_id() # Get new ID from tracker
                    v_id, consistent_type = self.tracker.update_track(
                        v_id, det_info['center'], frame_time_sec_abs, det_info['type']
                    )
                    processed_tracks_this_frame_map[v_id] = {
                        'box': det_info['box_coords'], 'type': consistent_type, 'id': v_id,
                        'center_for_zone': det_info['bottom_center'],
                        'event': None, 'status': 'new'
                    }
                    ids_seen_this_frame_custom.add(v_id)
            
            # --- Zone Transition Logic (Using data from processed_tracks_this_frame_map) ---
            ids_to_remove_this_frame = set()
            active_zone_keys = list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []

            for v_id, track_data in processed_tracks_this_frame_map.items():
                current_bbox_for_zone = track_data['box'] # This is the (x1,y1,x2,y2) from detection
                # For prev_pos, VehicleTracker.tracking_history stores center points usually
                # ZoneTracker.check_zone_transition expects prev_center_point and current_bbox
                prev_pos = None
                if v_id in self.tracker.tracking_history and len(self.tracker.tracking_history[v_id]) > 1:
                    prev_pos = self.tracker.tracking_history[v_id][-2] # This is a center point

                if prev_pos and self.zone_tracker and self.zone_tracker.zones:
                    if self.perf: self.perf.start_timer('zone_checking')
                    event_type, event_dir = self.zone_tracker.check_zone_transition(
                        prev_pos, current_bbox_for_zone, v_id, frame_time_sec_abs # Use current frame time (absolute from video start)
                    )
                    if self.perf: self.perf.end_timer('zone_checking')

                    v_state = self.tracker.active_vehicles.get(v_id) # State from VehicleTracker
                    consistent_type = track_data['type'] # Type determined by tracker

                    if event_type == "ENTRY":
                        # Your existing ENTRY and Inferred EXIT logic here, adapted slightly:
                        # - Use current_timestamp_abs for event times
                        # - Use consistent_type from track_data
                        # - Append to frame_log_entries
                        # - Update processed_tracks_this_frame_map[v_id]['event'] / ['status']
                        if v_state and v_state.get('status') == 'active': # Potentially an inferred exit
                            stored_entry_dir = v_state.get('entry_direction')
                            if stored_entry_dir and stored_entry_dir != event_dir and is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                final_type_inf = self.tracker.get_consistent_type(v_id, consistent_type) # Get final type before exit
                                success_inf, t_in_int_inf = self.tracker.register_exit(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'])
                                if success_inf:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_inf)
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': final_type_inf, 'event_type': 'EXIT', 'direction_from': stored_entry_dir, 'direction_to': event_dir, 'status': 'exited_inferred', 'time_in_intersection': t_in_int_inf, 'frame_number': frame_number_abs})
                                    ids_to_remove_this_frame.add(v_id)
                                    processed_tracks_this_frame_map[v_id]['event'] = 'EXIT' # Mark for drawing
                                # Now, register the new entry
                                if self.tracker.register_entry(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'], consistent_type):
                                    if self.perf: self.perf.record_vehicle_entry()
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': consistent_type, 'event_type': 'ENTRY', 'direction_from': event_dir, 'direction_to': None, 'status': 'entry', 'frame_number': frame_number_abs})
                                    processed_tracks_this_frame_map[v_id]['event'] = 'ENTRY' # Override for drawing
                                    processed_tracks_this_frame_map[v_id]['status'] = 'active'
                            # else: Re-entry or invalid inferred, handled by standard entry if needed
                        
                        # Standard new entry (or if inferred exit didn't apply and it's still not active)
                        if not (v_state and v_state.get('status') == 'active'): # If not already active from this entry dir
                            if self.tracker.register_entry(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'], consistent_type):
                                if self.perf: self.perf.record_vehicle_entry()
                                frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': consistent_type, 'event_type': 'ENTRY', 'direction_from': event_dir, 'direction_to': None, 'status': 'entry', 'frame_number': frame_number_abs})
                                processed_tracks_this_frame_map[v_id]['event'] = 'ENTRY'
                                processed_tracks_this_frame_map[v_id]['status'] = 'active'


                    elif event_type == "EXIT":
                        # Your existing EXIT logic here:
                        # - Use current_timestamp_abs, consistent_type
                        # - Append to frame_log_entries
                        # - Add to ids_to_remove_this_frame
                        # - Update processed_tracks_this_frame_map[v_id]['event'] / ['status']
                        if v_state and v_state.get('status') == 'active':
                            entry_dir_trk = v_state.get('entry_direction')
                            if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir, active_zone_keys):
                                final_exit_type = self.tracker.get_consistent_type(v_id, consistent_type)
                                success_ex, t_in_int_ex = self.tracker.register_exit(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'])
                                if success_ex:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_ex)
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': final_exit_type, 'event_type': 'EXIT', 'direction_from': entry_dir_trk, 'direction_to': event_dir, 'status': 'exit', 'time_in_intersection': t_in_int_ex, 'frame_number': frame_number_abs})
                                    ids_to_remove_this_frame.add(v_id)
                                    processed_tracks_this_frame_map[v_id]['event'] = 'EXIT'
                                    processed_tracks_this_frame_map[v_id]['status'] = 'exited'

            # --- Timeout Check (using self.tracker) ---
            timed_out_ids = self.tracker.check_timeouts(current_timestamp_abs)
            for v_id_to in timed_out_ids:
                ids_to_remove_this_frame.add(v_id_to)
                if self.perf: self.perf.record_vehicle_exit('timed_out')
                p_data = next((p for p in reversed(self.tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'), None)
                if p_data:
                     frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id_to, 'vehicle_type': p_data.get('type', 'UnknownType'), 'event_type': 'TIMEOUT', 'direction_from': p_data.get('entry_direction', 'UNKNOWN'), 'direction_to': 'TIMEOUT', 'status': 'timeout', 'time_in_intersection': p_data.get('time_in_intersection', 'N/A'), 'frame_number': frame_number_abs})
                if v_id_to in processed_tracks_this_frame_map:
                    processed_tracks_this_frame_map[v_id_to]['status'] = 'timed_out'


            # --- Update Misses & Remove data (using self.tracker) ---
            if self.tracker_type_internal == 'Custom':
                self.tracker.increment_misses(ids_seen_this_frame_custom)
            for v_id_rem in ids_to_remove_this_frame:
                self.tracker.remove_vehicle_data(v_id_rem)
                if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_rem)


            # --- Prepare data for GPU visualization ---
            current_active_vehicles_state = self.tracker.active_vehicles.copy()
            for v_id, data in processed_tracks_this_frame_map.items():
                if v_id not in ids_to_remove_this_frame: # Only visualize if still relevant
                    viz_status = 'detected' # Default
                    vehicle_state_for_viz = current_active_vehicles_state.get(v_id)
                    if vehicle_state_for_viz and vehicle_state_for_viz.get('status') == 'active':
                        viz_status = 'active'
                    elif data.get('event') == 'EXIT': # If explicitly exited this frame
                        viz_status = 'exiting'
                    elif data.get('event') == 'ENTRY' and not (vehicle_state_for_viz and vehicle_state_for_viz.get('status') == 'active'):
                         viz_status = 'entering' # New entry this frame

                    frame_viz_entries.append({
                        'box': data['box'], # Already scaled to model_input_size
                        'id': data['id'],
                        'type': data['type'],
                        'status': viz_status
                        # Add more if gpu_overlay.py needs it (e.g., trail points if you implement GPU trails)
                    })
            return frame_log_entries, frame_viz_entries
            # --- End process_single_frame_logic ---

        # --- Parallel Execution for the batch ---
        if self.perf: self.perf.start_timer('detection_processing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_frame_logic, i): i for i in range(batch_size_actual)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    frame_log, frame_viz_data = future.result()
                    batch_structured_log_data[idx] = frame_log
                    batch_detections_for_gpu_viz[idx] = frame_viz_data
                except Exception as exc:
                    print(f'\nError in post-processing frame {frame_numbers[idx]} (batch idx {idx}): {exc}')
                    import traceback; traceback.print_exc()
                    batch_structured_log_data[idx] = [] # Ensure lists are populated even on error
                    batch_detections_for_gpu_viz[idx] = []
        if self.perf: self.perf.end_timer('detection_processing')

        flat_log_data = [item for sublist in batch_structured_log_data for item in sublist]
        return flat_log_data, batch_detections_for_gpu_viz # batch_detections_for_gpu_viz is list of lists

    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        """
        Orchestrates video processing, handling chunking and consolidation.
        This function is now a GENERATOR, yielding status updates.
        The final yield will be a dict: {'final_message': "..."}
        """
        if not self.processing_lock.acquire(blocking=False):
            yield {'final_message': "Processing is already in progress."}
            return

        yield "Orchestrator: Initializing processing..."
        print(f"--- Orchestrator: Starting Video Processing ---"); print(f"Video: {video_path}")
        
        final_outcome_message = "Processing did not complete as expected." # Default

        # Ensure model is loaded (if not done in __init__ or if it needs re-checking)
        if self.model is None:
            yield "Orchestrator: Loading model..."
            print("Orchestrator: Model is None, attempting to load...")
            try:
                self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
                if self.model is None: raise RuntimeError("Model loading returned None in orchestrator.")
                self.model_names = getattr(self.model, 'names', {})
                yield "Orchestrator: Model loaded successfully."
            except Exception as e:
                error_msg = f"❌ FATAL: Error loading model: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}
                return

        def parse_dt(date_str, time_str): # Helper
            try:
                dt_str = f"{date_str}{time_str[:6]}"
                base_dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")
                return base_dt.replace(microsecond=int(time_str[6:].ljust(6, '0')))
            except Exception as e:
                print(f"Error parsing datetime: {date_str} {time_str} - {e}")
                return None
        
        original_start_dt = parse_dt(start_date_str, start_time_str)
        if not original_start_dt:
            error_msg = "❌ Error: Invalid start date/time format."
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}
            return

        yield "Orchestrator: Reading video properties..."
        duration_sec, _, _ = get_video_properties(video_path) # Assuming get_video_properties is robust
        if duration_sec is None:
            error_msg = f"❌ Error: Cannot read video properties: {video_path}"
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}
            return

        should_chunk = ENABLE_AUTO_CHUNKING and duration_sec > (AUTO_CHUNK_THRESHOLD_MINUTES * 60)
        chunk_files = [video_path] # Default to original file
        chunk_duration_sec_cfg = AUTO_CHUNK_DURATION_MINUTES * 60

        if should_chunk:
            status_msg_chunking = f"Orchestrator: Video duration ({duration_sec:.0f}s) triggers chunking (~{AUTO_CHUNK_DURATION_MINUTES} min chunks)."
            yield status_msg_chunking
            print(status_msg_chunking)
            
            # Define a simple callback for ffmpeg splitting status
            def ffmpeg_split_progress(progress_val, message_val): # progress_val not used yet by split_video_ffmpeg
                yield message_val # Yield the message from ffmpeg_split

            try:
                # This progress_callback for split_video_ffmpeg needs to be handled by that function
                # to yield messages if we want live ffmpeg progress (complex).
                # For now, split_video_ffmpeg has print statements and its own progress_callback parameter.
                # The one passed here from Gradio won't be directly used by ffmpeg's stdout.
                # Let's simplify: split_video_ffmpeg will print, and we yield before/after.
                yield "Orchestrator: Splitting video now... (this may take a while)"
                chunk_files = split_video_ffmpeg(video_path, CHUNK_TEMP_DIR, chunk_duration_sec_cfg, FFMPEG_PATH, progress_callback=None) # Pass None for now
                if not chunk_files: raise RuntimeError("FFmpeg splitting yielded no files.")
                yield f"Orchestrator: Video split into {len(chunk_files)} chunks."
            except Exception as e:
                error_msg = f"❌ Error during video splitting: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}
                return
            
            if os.path.exists(REPORT_TEMP_DIR): shutil.rmtree(REPORT_TEMP_DIR) # Clean old temp reports
            os.makedirs(REPORT_TEMP_DIR, exist_ok=True)
        else:
            status_msg_single = f"Orchestrator: Processing as single file (Duration: {duration_sec:.0f}s)."
            yield status_msg_single
            print(status_msg_single)

        all_chunk_processing_ok = True
        num_chunks = len(chunk_files)
        processed_chunk_results = [] # To store result_msg_part from each chunk

        for i, current_chunk_path in enumerate(chunk_files):
            chunk_start_offset = timedelta(seconds=i * chunk_duration_sec_cfg if should_chunk else 0)
            current_chunk_start_dt = original_start_dt + chunk_start_offset
            current_start_date_str_chunk = current_chunk_start_dt.strftime("%y%m%d")
            current_start_time_str_chunk = current_chunk_start_dt.strftime("%H%M%S") + f"{current_chunk_start_dt.microsecond // 1000:03d}"

            status_processing_chunk = f"Orchestrator: Starting segment {i+1}/{num_chunks}: {os.path.basename(current_chunk_path)}"
            yield status_processing_chunk
            print("*" * 60); print(status_processing_chunk)

            temp_report_override_path = None
            if should_chunk:
                base_chunk_name = os.path.splitext(os.path.basename(current_chunk_path))[0]
                temp_report_override_path = os.path.join(REPORT_TEMP_DIR, f"report_{base_chunk_name}.xlsx")

            # --- Define a callback for _process_single_video_file to yield its internal progress ---
            def single_file_progress_callback(p_value, status_message_internal):
                # This callback is called from within _process_single_video_file's loop
                # We need to yield this status message out to Gradio
                yield f"Chunk {i+1}/{num_chunks} - {status_message_internal}" # Prepend chunk info
            
            # _process_single_video_file is NOT a generator, so we can't loop over its yields here.
            # The progress_callback passed to it would need to use a queue or shared state if we
            # want live updates from its internal loop in Gradio without making it a generator too.
            # For now, the status updates from _process_single_video_file will primarily be to the console.
            # The yields here are for *orchestrator-level* status.
            
            result_msg_part = self._process_single_video_file(
                video_path=current_chunk_path,
                start_date_str=current_start_date_str_chunk,
                start_time_str=current_start_time_str_chunk,
                primary_direction=primary_direction,
                output_path_override=temp_report_override_path,
                progress_callback=None, # Pass None; _process_single_video_file prints console progress
                chunk_info=(i + 1, num_chunks)
            )
            processed_chunk_results.append(result_msg_part) # Store individual result

            if "❌ Error" in result_msg_part or "FATAL" in result_msg_part:
                all_chunk_processing_ok = False
                final_outcome_message = result_msg_part # Store the first critical error
                error_stop_msg = f"Orchestrator: Error processing {os.path.basename(current_chunk_path)}. Halting."
                yield error_stop_msg
                print(error_stop_msg)
                break # Stop processing further chunks
            else:
                yield f"Orchestrator: Finished segment {i+1}/{num_chunks}."


        print("*" * 60)
        # --- Final Outcome / Consolidation ---
        if all_chunk_processing_ok:
            if should_chunk:
                consolidation_start_msg = "Orchestrator: Consolidating reports..."
                yield consolidation_start_msg
                print(consolidation_start_msg)

                original_base = os.path.splitext(os.path.basename(video_path))[0]
                final_consolidated_report_path = os.path.join(REPORT_OUTPUT_DIR, f"detection_logs_{original_base}_CONSOLIDATED.xlsx")
                os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
                
                consolidated_path = consolidate_excel_reports(REPORT_TEMP_DIR, final_consolidated_report_path, original_start_dt.date())
                
                if consolidated_path:
                    # Combine individual chunk summaries for a more complete message
                    full_summary = "Chunk Processing Summaries:\n" + "\n\n".join(processed_chunk_results)
                    full_summary += f"\n\n✅ Processing completed.\nConsolidated report: {consolidated_path}"
                    final_outcome_message = full_summary
                    yield "Orchestrator: Report consolidation successful."
                    # Optional: Clean up temp report dir
                    # shutil.rmtree(REPORT_TEMP_DIR)
                else:
                    final_outcome_message = f"⚠️ Processing finished, but report consolidation failed. Chunk reports in {REPORT_TEMP_DIR}."
                    yield "Orchestrator: Report consolidation failed."
            else: # Single file success
                final_report_path_single = None
                # Use the result from the single processed file
                result_msg_part_single = processed_chunk_results[0] if processed_chunk_results else ""
                
                match = re.search(r"Report segment: (.+\.xlsx)", result_msg_part_single) or \
                        re.search(r"Excel report: (.+\.xlsx)", result_msg_part_single)
                if match: final_report_path_single = match.group(1)

                if final_report_path_single and os.path.exists(final_report_path_single):
                    # For single file, the result_msg_part_single already contains the full summary.
                    final_outcome_message = result_msg_part_single
                    # No need to add "Report: ..." as it's in result_msg_part_single
                else:
                    final_outcome_message = f"⚠️ Processing finished for single file, but report file not found. Message: {result_msg_part_single}"

        elif not final_outcome_message: # Error occurred, but message wasn't set by the loop
            final_outcome_message = "❌ Processing stopped due to errors in chunk processing."

        if should_chunk:
            cleanup_msg = "Orchestrator: Cleaning up temporary video chunks..."
            yield cleanup_msg
            print(cleanup_msg)
            try:
                shutil.rmtree(CHUNK_TEMP_DIR)
                yield "Orchestrator: Temporary video chunks cleaned up."
            except Exception as e:
                warn_cleanup_msg = f"Warning: Could not remove temp chunk dir: {e}"
                yield warn_cleanup_msg
                print(warn_cleanup_msg)

        release_msg = "Orchestrator: All processing stages finished. Releasing lock."
        yield release_msg
        print(release_msg)
        if self.processing_lock.locked(): self.processing_lock.release()

        # The VERY LAST yield is the final package for Gradio's handle_process
        yield {'final_message': final_outcome_message}

    # --- Core Processing Logic for ONE file/chunk ---
    def _process_single_video_file(self, video_path, start_date_str, start_time_str, primary_direction, output_path_override=None, progress_callback=None, chunk_info=(1,1)):
        """ Processes a single video file (original or chunk) using Nvidia Reader/Writer. """
        print(f"_process_single_video_file: Initializing for {os.path.basename(video_path)}")
        # --- State Reset ---
        if self.tracker_type_internal == 'Custom':
            self.tracker = VehicleTracker()
        # Add other tracker initializations here if TRACKER_TYPE changes
        # else: raise NotImplementedError(f"Tracker type {self.tracker_type_internal} not fully implemented for reset.")
        self.zone_tracker = None
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        self.writer_instance = None
        self.reader_thread = None
        self.final_output_path_current_file = None # For this chunk's video output

        try:
            # --- Parse Start Datetime for this chunk/file ---
            dt_str = f"{start_date_str}{start_time_str[:6]}"; base_dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")
            start_datetime_chunk = base_dt.replace(microsecond=int(start_time_str[6:].ljust(6, '0')))

            # --- Zone Setup ---
            if self.line_mode == 'hardcoded':
                # ... (your existing hardcoded zone loading logic from original process_video) ...
                 if not LINE_POINTS: raise ValueError("Config LINE_POINTS missing")
                 valid_polys = {k: v for k, v in LINE_POINTS.items() if isinstance(v, list) and len(v) >= 3}
                 if not valid_polys: raise ValueError("No valid polygons in config LINE_POINTS")
                 self.detection_zones_polygons = {k: np.array(v, dtype=np.int32) for k, v in valid_polys.items()}
            elif self.line_mode == 'interactive':
                # ... (your existing interactive zone loading logic using self.gradio_polygons) ...
                 if not self.gradio_polygons: raise ValueError("Gradio polygons not defined")
                 valid_polys = {k: v for k, v in self.gradio_polygons.items() if v and len(v) >= 3}
                 if len(valid_polys) < 2: raise ValueError("Need >= 2 valid Gradio zones")
                 self.detection_zones_polygons = {k: np.array(v, dtype=np.int32) for k, v in valid_polys.items()}
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("ZoneTracker init failed")
            print(f"ZoneTracker initialized with {len(self.zone_tracker.zones)} zones for {os.path.basename(video_path)}.")


            # --- Get Video Properties for the current file ---
            duration_sec_file, original_fps_file, total_frames_file = get_video_properties(video_path)
            if duration_sec_file is None: raise FileNotFoundError(f"Cannot open or read properties: {video_path}")
            if original_fps_file <= 0: original_fps_file = 30.0

            # --- Setup Reader Thread (NvidiaFrameReader) ---
            self.frame_read_queue = queue.Queue(maxsize=self.batch_size * 4)
            # Assuming NvidiaFrameReader is globally imported or in self
            self.reader_instance = NvidiaFrameReader(
                video_path, self.target_fps, original_fps_file,
                self.frame_read_queue, self.stop_event, device_id=0 # Assuming GPU 0
            )
            # Wait for reader to initialize and get dims (important for writer)
            if not self.reader_instance._initialize_decoder(): # Call protected method - might need refactor
                raise RuntimeError("NvidiaReader failed to initialize its decoder.")
            reader_w, reader_h = self.reader_instance.width, self.reader_instance.height
            self.reader_thread = threading.Thread(target=self.reader_instance.run, daemon=True); self.reader_thread.start()

            # --- Setup Writer Thread (NvidiaFrameWriter if GPU_VIZ_ENABLED) ---
            _viz_enabled_this_run = ENABLE_VISUALIZATION and GPU_VIZ_ENABLED
            if _viz_enabled_this_run:
                # Define final output path for THIS chunk's video
                chunk_video_base = f"output_{os.path.splitext(os.path.basename(video_path))[0]}"
                final_muxed_path = os.path.join(VIDEO_OUTPUT_DIR, f"{chunk_video_base}{FINAL_VIDEO_EXTENSION}")
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

                self.writer_instance = NvidiaFrameWriter(
                    output_path=final_muxed_path, width=reader_w, height=reader_h,
                    fps=self.target_fps, # Output video at target FPS
                    codec=ENCODER_CODEC, bitrate=ENCODER_BITRATE, preset=ENCODER_PRESET,
                    temp_dir=CHUNK_TEMP_DIR, # Place raw stream in temp chunk dir
                    device_id=0)
                self.writer_instance.start()
                print("Nvidia Writer thread started.")
            else:
                self.writer_instance = None
                print("GPU Visualization/Writing disabled for this run.")

            # --- Main Processing Loop ---
            all_detections_structured_log = [] # For this file's report
            frames_processed_count = 0
            if self.perf: self.perf.start_processing(total_frames_file)

            frames_batch_gpu_tensors = [] # List to hold GPU tensors from reader
            frame_numbers_batch_abs = []  # Absolute frame numbers from original video
            frame_times_batch_abs = []    # Absolute time offsets from original video start

            last_progress_print_time = time.monotonic()
            current_chunk_idx_disp, total_chunks_disp = chunk_info # For display

            while True: # Main processing loop
                try:
                    frame_data = self.frame_read_queue.get(timeout=120) # Increased timeout
                    if frame_data is None: self.frame_read_queue.task_done(); break # Reader done
                    
                    gpu_tensor, frame_num_abs, frame_time_abs = frame_data
                    if gpu_tensor is None: self.frame_read_queue.task_done(); continue

                    frames_batch_gpu_tensors.append(gpu_tensor)
                    frame_numbers_batch_abs.append(frame_num_abs)
                    frame_times_batch_abs.append(frame_time_abs)

                    if len(frames_batch_gpu_tensors) >= self.batch_size:
                        batch_start_mono = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch_processed = self._preprocess_batch(frames_batch_gpu_tensors) # Input for YOLO
                        if self.perf: self.perf.end_timer('preprocessing')

                        if input_batch_processed is None or input_batch_processed.nelement() == 0:
                            # print(f"Warning: Empty batch after preprocessing. Frame numbers: {frame_numbers_batch_abs}")
                            frames_batch_gpu_tensors.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                            # Mark tasks done for consumed items
                            for _ in range(len(frames_batch_gpu_tensors)): # This was old len, use new one
                                try: self.frame_read_queue.task_done()
                                except ValueError: break
                            continue

                        # --- Inference ---
                        stream = None # Handle self.streams if PARALLEL_STREAMS > 1
                        if self.device == 'cuda' and self.streams: stream = self.streams[self.current_stream_index % len(self.streams)]; self.current_stream_index += 1
                        
                        with torch.cuda.stream(stream) if stream else torch.no_grad():
                            with torch.amp.autocast(device_type=self.device, enabled=MIXED_PRECISION and self.device == 'cuda'):
                                if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream)
                                yolo_results = self.model(input_batch_processed, verbose=False)
                                if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                        if stream: stream.synchronize()
                        if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                        # --- Post-processing ---
                        # Pass start_datetime_chunk which is absolute time for this chunk
                        batch_log_entries, batch_viz_data_for_gpu = self._process_inference_results(
                            yolo_results, input_batch_processed, frame_numbers_batch_abs, frame_times_batch_abs, start_datetime_chunk
                        )
                        all_detections_structured_log.extend(batch_log_entries)
                        if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in yolo_results if hasattr(r,'boxes') and r.boxes))

                        # --- GPU Visualization & Writing ---
                        if _viz_enabled_this_run and self.writer_instance:
                            if self.perf: self.perf.start_timer('drawing_gpu')
                            # input_batch_processed is [B,C,H,W], float, normalized [0,1]
                            # add_gpu_overlays should expect this format
                            visualized_batch = add_gpu_overlays(
                                input_batch_processed, batch_viz_data_for_gpu, self.zone_tracker.zones
                            )
                            if self.perf: self.perf.end_timer('drawing_gpu')

                            if visualized_batch is not None:
                                for frame_tensor_to_write in visualized_batch: # NvidiaWriter takes single tensors
                                    # Convert format for encoder if needed (e.g., float [0,1] -> uint8 [0,255])
                                    # PyNvEncoder might need specific dtype. Let's assume it takes float for now or needs conversion.
                                    # Example: frame_to_encode = (frame_tensor_to_write.clamp(0,1) * 255.0).byte()
                                    self.writer_instance.put(frame_tensor_to_write) # Pass tensor directly
                        
                        # --- Stats, Cleanup, Progress ---
                        batch_time_taken = time.monotonic() - batch_start_mono
                        frames_processed_count += len(frames_batch_gpu_tensors)
                        if self.perf: self.perf.record_batch_processed(len(frames_batch_gpu_tensors), batch_time_taken); self.perf.sample_system_metrics()
                        
                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                             if self.perf: self.perf.start_timer('memory_cleanup')
                             last_frame_time_in_batch = frame_times_batch_abs[-1] if frame_times_batch_abs else 0
                             cleanup_tracking_data(self.tracker, self.zone_tracker, last_frame_time_in_batch)
                             cleanup_memory()
                             self.last_cleanup_time = current_mono_time
                             if self.perf: self.perf.end_timer('memory_cleanup')
                        
                        # Progress Callback for Gradio
                        if progress_callback and self.perf:
                            p_stats = self.perf.get_progress()
                            base_prog = (current_chunk_idx_disp - 1) / total_chunks_disp
                            chunk_prog = p_stats.get('percent', 0) / 100.0
                            overall = min(1.0, base_prog + (chunk_prog / total_chunks_disp))
                            status = f"Chunk {current_chunk_idx_disp}/{total_chunks_disp}: {p_stats['percent']:.1f}% (FPS:{p_stats['fps']:.1f}|ETA:{p_stats['eta']})"
                            try: progress_callback(overall, status)
                            except Exception as cb_err: print(f"Warning: Gradio progress_callback error: {cb_err}")

                        # Console Progress
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0):
                             prog_stats_console = self.perf.get_progress()
                             active_trk_count = self.tracker.get_active_vehicle_count() if hasattr(self.tracker, 'get_active_vehicle_count') else 'N/A'
                             print(f"\rChunk {current_chunk_idx_disp}/{total_chunks_disp} Prog:{prog_stats_console['percent']:.1f}%|FPS:{prog_stats_console['fps']:.1f}|Active:{active_trk_count}|ETA:{prog_stats_console['eta']} ", end="")
                             last_progress_print_time = current_mono_time

                        # Clear Batch lists
                        frames_batch_gpu_tensors.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                        del input_batch_processed, yolo_results, batch_log_entries, batch_viz_data_for_gpu
                        if _viz_enabled_this_run and self.writer_instance: del visualized_batch
                        cleanup_memory()

                    self.frame_read_queue.task_done()
                except queue.Empty:
                    print("\n_process_single_video_file: Frame reader queue timed out. File might be shorter than expected or reader stuck."); break
                except Exception as e:
                    print(f"\n_process_single_video_file: ERROR during processing loop: {e}")
                    import traceback; traceback.print_exc()
                    if self.stop_event: self.stop_event.set(); # Signal other threads
                    break
            # --- End Main Processing Loop ---

            # --- Process Final Batch ---
            if frames_batch_gpu_tensors:
                print("\n_process_single_video_file: Processing final batch...")
                # ... (repeat batch processing logic for remaining frames) ...
                # This needs to be a copy of the logic inside "if len(frames_batch_gpu_tensors) >= self.batch_size:"
                # Ensure to add frames_processed_count += len(frames_batch_gpu_tensors)
                if self.perf: self.perf.start_timer('preprocessing')
                input_batch_processed = self._preprocess_batch(frames_batch_gpu_tensors)
                if self.perf: self.perf.end_timer('preprocessing')
                if input_batch_processed and input_batch_processed.nelement() > 0:
                    # ... (Inference, Post-processing, GPU Viz, Write, Stats) ...
                    pass # Placeholder for full final batch logic
                frames_processed_count += len(frames_batch_gpu_tensors)
                del frames_batch_gpu_tensors # etc.
                cleanup_memory()

            # --- Finalize (Forced Exits, Thread Joins, Report) ---
            if self.tracker and hasattr(self.tracker, 'active_vehicles') and self.tracker.active_vehicles:
                 print(f"Forcing exit for {len(self.tracker.active_vehicles)} remaining vehicles...")
                 # Use last known timestamp from this chunk processing
                 last_ts_in_chunk = start_datetime_chunk + timedelta(seconds=frame_times_batch_abs[-1]) if frame_times_batch_abs else start_datetime_chunk
                 # ... (Implement your forced exit logic from original process_video, append to all_detections_structured_log) ...
                 for v_id_force in list(self.tracker.active_vehicles.keys()):
                     if self.tracker.force_exit_vehicle(v_id_force, last_ts_in_chunk):
                         if self.perf: self.perf.record_vehicle_exit('forced_exit')
                         # Find completed path entry for log
                         p_data = next((p for p in reversed(self.tracker.completed_paths) if p['id']==v_id_force and p['status']=='forced_exit'), None)
                         if p_data:
                              all_detections_structured_log.append({'timestamp_dt': last_ts_in_chunk, 'vehicle_id': v_id_force, 'vehicle_type': p_data.get('type'), 'event_type':'FORCED_EXIT', 'direction_from': p_data.get('entry_direction','?'), 'status':'forced_exit', 'time_in_intersection': p_data.get('time_in_intersection')})


            print(f"_process_single_video_file: Signaling threads to stop for {os.path.basename(video_path)}...");
            if self.stop_event: self.stop_event.set()
            if self.writer_instance: self.writer_instance.put(None) # EOS for writer

            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=10)
            
            mux_success_this_file = True
            if self.writer_instance:
                 mux_success_this_file = self.writer_instance.stop() # This blocks & runs muxing
                 self.final_output_path_current_file = self.writer_instance.output_path if mux_success_this_file else None

            # --- Performance Summary ---
            total_time_taken_file=0.0; fps_file=0.0; completed_paths_count_valid_file=0
            if self.perf: self.perf.end_processing(); self.perf.print_summary(); total_time_taken_file=self.perf.total_time; fps_file=frames_processed_count/total_time_taken_file if total_time_taken_file>0 else 0
            
            final_completed_paths_for_report = []
            if self.tracker:
                 if hasattr(self.tracker, 'get_completed_paths'): final_completed_paths_for_report = self.tracker.get_completed_paths()
                 elif hasattr(self.tracker, 'completed_paths'): final_completed_paths_for_report = self.tracker.completed_paths
            
            for path_rep in final_completed_paths_for_report: # Calculate valid for summary
                if path_rep.get('status') == 'exited' and path_rep.get('entry_direction') and path_rep.get('exit_direction') and \
                   path_rep.get('exit_direction') not in ['TIMEOUT', 'FORCED', 'UNKNOWN', None]:
                    completed_paths_count_valid_file +=1
            
            # --- Create report for THIS CHUNK/FILE ---
            print(f"Generating report for {os.path.basename(video_path)}...")
            excel_file_path = create_excel_report(
                completed_paths_data=final_completed_paths_for_report,
                start_datetime=start_datetime_chunk, # Use the absolute start time of THIS chunk
                primary_direction=primary_direction,
                video_path=video_path, # Original path or chunk path
                output_path_override=output_path_override
            )

            result_msg_this_file = f"✅ Processing completed for {os.path.basename(video_path)}.\n"
            if _viz_enabled_this_run:
                 if self.final_output_path_current_file and mux_success_this_file: result_msg_this_file += f"Output video segment: '{self.final_output_path_current_file}'.\n"
                 elif not mux_success_this_file : result_msg_this_file += f"Output video segment generation failed (muxing error).\n"
            if excel_file_path: result_msg_this_file += f"Report segment: {excel_file_path}\n"
            else: result_msg_this_file += f"Report segment generation failed for {os.path.basename(video_path)}.\n"
            result_msg_this_file += f"--- Summary Stats for {os.path.basename(video_path)} ---\nSTAT_FRAMES_PROCESSED={frames_processed_count}\nSTAT_TIME_SECONDS={total_time_taken_file:.2f}\nSTAT_FPS={fps_file:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid_file}\n--- End Stats ---"
            
            return result_msg_this_file

        except Exception as e_outer:
            print(f"\nFATAL ERROR during _process_single_video_file for {os.path.basename(video_path)}: {e_outer}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            if hasattr(self, 'writer_instance') and self.writer_instance:
                try: self.writer_instance.stop() # Attempt to stop writer and mux/cleanup
                except: pass
            return f"❌ Error processing {os.path.basename(video_path)}: {e_outer}"
        finally:
            # --- Cleanup per-file resources ---
            if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1)
            # NvidiaWriter.stop() should have been called and joined
            self.tracker=None; self.zone_tracker=None; self.perf=None;
            self.frame_read_queue=None; self.video_write_queue=None;
            self.reader_thread=None; self.writer_instance=None; self.stop_event=None;
            self.final_output_path_current_file = None
            cleanup_memory()
            print(f"Finished cleanup for {os.path.basename(video_path)}")

# --- End of VideoProcessor class ---