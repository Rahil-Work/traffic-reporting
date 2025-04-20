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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, DEBUG_MODE, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, # LINE_POINTS should now contain polygon vertices lists
    MODEL_INPUT_SIZE
)

# Modules
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
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
        self.line_mode = LINE_MODE # Controls zone definition method
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
        self.vehicle_tracker = None
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None # Holds final polygons (NumPy arrays)
        self.model_input_size_config = MODEL_INPUT_SIZE
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
                try: self.frame_read_queue.put(None, timeout=1.0)
                except queue.Full: pass

    def _frame_writer_task(self, output_path):
        out = None; writer_initialized = False; frames_written = 0
        actual_output_path = output_path
        try:
             fourcc = cv2.VideoWriter_fourcc(*'mp4v');
             out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
             if not out.isOpened():
                 print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI.");
                 avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                 fourcc = cv2.VideoWriter_fourcc(*'XVID');
                 out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                 if out.isOpened(): actual_output_path = avi_path; self.final_output_path = avi_path
                 else: print(f"ERROR: Writer failed to open for both MP4 and AVI."); return
             writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")
             while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0)
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
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e: print(f"ERROR in writer thread: {e}")
        finally:
            if out and writer_initialized: out.release(); debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            if self.video_write_queue: self.video_write_queue.put(None)


    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frame_list):
        tensors = []; target_size = self.model_input_size_config
        for frame in frame_list:
            if frame is None or not isinstance(frame, np.ndarray): continue
            img=cv2.resize(frame,(target_size,target_size)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=img.transpose(2,0,1); img=np.ascontiguousarray(img); tensors.append(img)
        if not tensors: return None
        batch_tensor=torch.from_numpy(np.stack(tensors)).to(self.device)
        dtype=torch.float16 if (MIXED_PRECISION and self.device=='cuda') else torch.float32
        batch_tensor = batch_tensor.to(dtype)/255.0; return batch_tensor

    def _process_inference_results(self, results, frames_batch, frame_numbers, frame_times, start_datetime):
        batch_detections_list = [[] for _ in range(len(results))]
        processed_frames_list = [None] * len(results)
        tracker_lock = self.tracker_lock

        def process_single_frame(idx):
            result=results[idx]; original_frame=frames_batch[idx]; frame_number=frame_numbers[idx]
            frame_time_sec=frame_times[idx]; current_timestamp=start_datetime+timedelta(seconds=frame_time_sec)
            output_frame=cv2.resize(original_frame.copy(), self.frame_shape); frame_local_detections=[]
            if not self.vehicle_tracker or not self.zone_tracker: return [], output_frame

            detections_this_frame = []; ids_processed_this_frame_map = {}
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls_id=int(box.cls[0]); v_type_name=result.names.get(cls_id,f"CLS_{cls_id}")
                    x1,y1,x2,y2=box.xyxy[0].cpu().numpy(); model_h,model_w=result.orig_shape
                    out_h,out_w=self.frame_shape[1],self.frame_shape[0]
                    x1o=max(0,int(x1*out_w/model_w)); y1o=max(0,int(y1*out_h/model_h)); x2o=min(out_w-1,int(x2*out_w/model_w)); y2o=min(out_h-1,int(y2*out_h/model_h))
                    center_pt=((x1o+x2o)//2, (y1o+y2o)//2)
                    detections_this_frame.append({'box_coords':(x1o,y1o,x2o,y2o), 'center':center_pt, 'type':v_type_name})

            ids_seen_in_lock = set(); ids_to_remove_this_frame = set()
            with tracker_lock:
                if self.perf: self.perf.start_timer('tracking_update')
                for det_info in detections_this_frame:
                    center_point, v_type_name = det_info['center'], det_info['type']
                    matched_id = self.vehicle_tracker.find_best_match(center_point, frame_time_sec)
                    v_id, c_type = self.vehicle_tracker.update_track(matched_id if matched_id else self.vehicle_tracker._get_next_id(), center_point, frame_time_sec, v_type_name)
                    ids_processed_this_frame_map[v_id] = {'center':center_point, 'box':det_info['box_coords'], 'type':c_type, 'event':None, 'status':'detected'}
                    ids_seen_in_lock.add(v_id)

                for v_id in list(ids_processed_this_frame_map.keys()):
                    current_data = ids_processed_this_frame_map[v_id]
                    center_point = current_data['center']
                    current_bbox_coords = current_data['box']
                    c_type = current_data['type']
                    prev_pos = None
                    if v_id in self.vehicle_tracker.tracking_history and len(self.vehicle_tracker.tracking_history[v_id]) > 1:
                        prev_pos = self.vehicle_tracker.tracking_history[v_id][-2]

                    if prev_pos and self.zone_tracker and self.zone_tracker.zones:
                        if self.perf: self.perf.start_timer('zone_checking')
                        event_type, event_dir = self.zone_tracker.check_zone_transition(
                            prev_pos, current_bbox_coords, v_id, frame_time_sec
                        )
                        if self.perf: self.perf.end_timer('zone_checking')

                        if event_type == "ENTRY":
                            v_state = self.vehicle_tracker.active_vehicles.get(v_id)
                            # Check if vehicle was already active coming from a *different* zone
                            if v_state and v_state.get('status') == 'active':
                                stored_entry_dir = v_state.get('entry_direction')
                                if stored_entry_dir and stored_entry_dir != event_dir:
                                    # --- INFERRED EXIT: Entered new zone (event_dir) while active from stored_entry_dir ---
                                    print(f"INFERRED EXIT: V:{v_id} entered '{event_dir}' while active from '{stored_entry_dir}'. Path complete.")
                                    active_zone_keys = list(self.zone_tracker.zones.keys())
                                    if is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                        entry_time_stored = v_state.get('entry_time')
                                        time_in_intersection = (current_timestamp - entry_time_stored).total_seconds() if entry_time_stored else None
                                        # Manually add to completed paths
                                        completed_path_data = {
                                            'id': v_id, 'type': c_type, # Use current type? Or type from v_state? Using current for now.
                                            'entry_direction': stored_entry_dir,
                                            'exit_direction': event_dir, # Exited towards the newly entered zone
                                            'entry_time': entry_time_stored,
                                            'exit_time': current_timestamp,
                                            'status': 'exited', # Mark as valid exit
                                            'time_in_intersection': round(time_in_intersection, 2) if time_in_intersection is not None else None
                                        }
                                        self.vehicle_tracker.completed_paths.append(completed_path_data)
                                        # Log the inferred exit event
                                        d_str, t_str = format_timestamp(current_timestamp)
                                        frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{c_type} EXITED TO {event_dir.upper()} (FROM {stored_entry_dir.upper()}) [Inferred]", 'frame_number':frame_number, 'vehicle_id':v_id, 'status':'exit', 'time_in_intersection':time_in_intersection})
                                        # Record performance
                                        if self.perf: self.perf.record_vehicle_exit('exited', time_in_intersection)
                                        # Mark for removal from active tracking
                                        ids_to_remove_this_frame.add(v_id)
                                        ids_processed_this_frame_map[v_id]['event'] = 'EXIT' # Mark event for drawing
                                    else:
                                        # Invalid movement detected (e.g., U-turn inferred) - treat as new entry? Or log error?
                                        print(f"INFO: V:{v_id} entered '{event_dir}' from '{stored_entry_dir}', but movement invalid by config. Treating as new entry.")
                                        # Fall through to register as a new entry from event_dir
                                        # Reset the state by removing first? This might cause issues.
                                        # Safest is likely to just force-exit the old track and start a new one if invalid.
                                        # OR: just ignore the invalid inferred path and don't register the new entry?
                                        # Let's ignore the invalid path and do nothing further for this v_id this frame.
                                        pass # Do not register exit or new entry if inferred path is invalid


                                else:
                                    # Entered same zone again or wasn't active properly - register normally
                                    if self.vehicle_tracker.register_entry(v_id, event_dir, current_timestamp, center_point, c_type):
                                        if self.perf: self.perf.record_vehicle_entry()
                                        d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{c_type} ENTERED FROM {event_dir.upper()}", 'frame_number':frame_number, 'vehicle_id':v_id, 'status':'entry'})
                                        ids_processed_this_frame_map[v_id]['event'] = 'ENTRY'
                            else:
                                # Vehicle wasn't active - standard new entry
                                if self.vehicle_tracker.register_entry(v_id, event_dir, current_timestamp, center_point, c_type):
                                    if self.perf: self.perf.record_vehicle_entry()
                                    d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{c_type} ENTERED FROM {event_dir.upper()}", 'frame_number':frame_number, 'vehicle_id':v_id, 'status':'entry'})
                                    ids_processed_this_frame_map[v_id]['event'] = 'ENTRY'

                        elif event_type == "EXIT":
                            # --- Keep Explicit EXIT Logic (Handles direct boundary crossing detection) ---
                            entry_dir_trk = self.vehicle_tracker.active_vehicles.get(v_id, {}).get('entry_direction')
                            active_zone_keys = list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []
                            is_valid = is_valid_movement(entry_dir_trk, event_dir, active_zone_keys)
                            # print(f"DEBUG_EXIT_VALIDATION: V:{v_id} EXPLICIT EXIT Event Dir:{event_dir} | EntryDir:{entry_dir_trk} | IsValid?:{is_valid}") # Optional Debug
                            if entry_dir_trk and is_valid:
                                # Use center_point or current_bottom_center for exit point registration? Using center_point for now.
                                success, t_in_int = self.vehicle_tracker.register_exit(v_id, event_dir, current_timestamp, center_point)
                                if success:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int)
                                    d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{c_type} EXITED TO {event_dir.upper()} (FROM {entry_dir_trk.upper()})", 'frame_number':frame_number, 'vehicle_id':v_id, 'status':'exit', 'time_in_intersection':t_in_int})
                                    ids_to_remove_this_frame.add(v_id)
                                    ids_processed_this_frame_map[v_id]['event'] = 'EXIT'

                timed_out_ids = self.vehicle_tracker.check_timeouts(current_timestamp)
                for v_id_to in timed_out_ids:
                    ids_to_remove_this_frame.add(v_id_to);
                    if self.perf: self.perf.record_vehicle_exit('timed_out')
                    p_data = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'), None)
                    if p_data: d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{p_data['type']} TIMED OUT (FROM {p_data.get('entry_direction','UNKNOWN').upper()})", 'frame_number':frame_number, 'vehicle_id':v_id_to, 'status':'timeout', 'time_in_intersection':p_data.get('time_in_intersection','N/A')})
                    if v_id_to in ids_processed_this_frame_map: ids_processed_this_frame_map[v_id_to]['status']='timed_out'

                self.vehicle_tracker.increment_misses(ids_seen_in_lock)
                for v_id_rem in ids_to_remove_this_frame:
                    self.vehicle_tracker.remove_vehicle_data(v_id_rem)
                    if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_rem)
                if self.perf: self.perf.end_timer('tracking_update')
            # --- End Lock ---

            # --- Drawing Pass ---
            if self.perf: self.perf.start_timer('drawing')
            if self.zone_tracker and self.zone_tracker.zones: output_frame = visual_overlay.draw_zones(output_frame, self.zone_tracker)
            active_vehicle_snapshot = self.vehicle_tracker.active_vehicles.copy()
            for v_id_draw, data in ids_processed_this_frame_map.items():
                center_pt_draw, box_draw, c_type_draw = data['center'], data['box'], data['type']
                current_draw_status = data.get('status', 'detected'); display_status="detected"; entry_dir_draw=None; t_active_draw=None
                if v_id_draw in active_vehicle_snapshot:
                     v_state = active_vehicle_snapshot[v_id_draw]
                     if v_state.get('status') == 'active':
                         display_status='active'; entry_dir_draw=v_state.get('entry_direction')
                         t_active_draw=(current_timestamp - v_state['entry_time']).total_seconds() if v_state.get('entry_time') else None
                if current_draw_status != 'timed_out':
                    visual_overlay.draw_detection_box(output_frame, box_draw, v_id_draw, c_type_draw, status=display_status, entry_dir=entry_dir_draw, time_active=t_active_draw)
                trail = self.vehicle_tracker.get_tracking_trail(v_id_draw); visual_overlay.draw_tracking_trail(output_frame, trail, v_id_draw)
                event_marker = data.get('event');
                if event_marker: visual_overlay.draw_event_marker(output_frame, center_pt_draw, event_marker, v_id_draw[-4:])
            output_frame = visual_overlay.add_status_overlay(output_frame, frame_number, current_timestamp, self.vehicle_tracker)
            if self.perf: self.perf.end_timer('drawing')
            return frame_local_detections, output_frame
            # --- End process_single_frame ---

        if self.perf: self.perf.start_timer('detection_processing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
             futures = {executor.submit(process_single_frame, i): i for i in range(len(results))}
             for future in concurrent.futures.as_completed(futures):
                 idx = futures[future]
                 try:
                     frame_detections, processed_frame = future.result()
                     batch_detections_list[idx] = frame_detections; processed_frames_list[idx] = processed_frame
                 except Exception as exc:
                     print(f'Frame {idx} failed: {exc}'); import traceback; traceback.print_exc()
                     processed_frames_list[idx] = cv2.resize(frames_batch[idx].copy(), self.frame_shape) # Fallback frame
        final_batch_detections = [det for frame_list in batch_detections_list for det in frame_list]
        final_processed_frames = processed_frames_list
        if self.perf: self.perf.end_timer('detection_processing')
        return final_batch_detections, final_processed_frames
    # --- End _process_inference_results ---


    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str):
        if not self.processing_lock.acquire(blocking=False):
            print("Warning: Processing lock already acquired."); return "Processing is already in progress."

        self.vehicle_tracker = VehicleTracker(); self.zone_tracker = None # Reset
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event(); self.frame_read_queue = queue.Queue(maxsize=self.batch_size*4)
        self.video_write_queue = queue.Queue(maxsize=self.batch_size*4); self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None; output_path = None; self.final_output_path = None

        print(f"--- Starting New Video Processing Run ---"); print(f"Video: {video_path}"); print(f"Timestamp: {start_date_str} {start_time_str}"); print(f"Zone Mode: {self.line_mode.upper()}")
        try:
            # --- Determine and Validate Zones (Polygon Mode) ---
            self.detection_zones_polygons = None # Ensure reset
            if self.line_mode == 'hardcoded':
                # ** Load and validate POLYGONS from config.py **
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded mode: LINE_POINTS not found/empty in config.py.")
                valid_polygons_hc = { dir: pts for dir, pts in LINE_POINTS.items() if isinstance(pts, list) and len(pts) >= 3 and all(isinstance(pt, tuple) and len(pt)==2 for pt in pts) }
                if not valid_polygons_hc: raise ValueError("Hardcoded mode: No valid polygons found in LINE_POINTS in config.py")
                if len(valid_polygons_hc) < len(LINE_POINTS): print("Warning: Some hardcoded LINE_POINTS ignored (invalid format/points).")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons_hc.items() }
                print(f"Using {len(self.detection_zones_polygons)} hardcoded zones.")
                if len(self.detection_zones_polygons) < 2: print("Warning: Fewer than 2 valid hardcoded zones.")

            elif self.line_mode == 'interactive':
                print(f"DEBUG: Gradio Polygons at start of process: {self.gradio_polygons}")
                if not self.gradio_polygons: raise ValueError("Interactive mode: No zone polygons defined.")
                valid_polygons = { dir: pts for dir, pts in self.gradio_polygons.items() if pts and len(pts) >= 3 }
                print(f"DEBUG: Valid Gradio Polygons (>=3 points): {valid_polygons}")
                if len(valid_polygons) < 2: raise ValueError(f"Interactive mode: Need >= 2 valid zones. Found {len(valid_polygons)}.")
                if len(valid_polygons) < len(self.gradio_polygons): print("Warning: Some drawn polygons ignored (< 3 points).")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons.items() }
                print(f"DEBUG: Final detection_zones_polygons to be used: {self.detection_zones_polygons}")
                print(f"Using {len(self.detection_zones_polygons)} valid zones from Gradio.")
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")

            # --- Initialize ZoneTracker ---
            if not self.detection_zones_polygons: raise ValueError("Zone polygon data missing before initializing ZoneTracker.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones:
             # Raise a runtime error if zones did not get created successfully
             raise RuntimeError("CRITICAL ERROR: ZoneTracker failed to initialize with valid zones. Cannot proceed.")

            # --- Init Model, Time, Video ---
            if video_path is None: raise ValueError("Video path missing.")
            if self.model is None: self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model {MODEL_PATH}")
            try: # Parse time
                start_date=datetime.strptime(start_date_str, "%y%m%d"); time_part,ms_part=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"0")
                base_time=datetime.strptime(time_part,"%H%M%S").time(); microseconds=int(ms_part.ljust(3,'0'))*1000
                start_time=base_time.replace(microsecond=microseconds); start_datetime=datetime.combine(start_date.date(), start_time)
                print(f"Video Start Timestamp: {start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e: raise ValueError(f"Invalid date/time format: {e}.")
            cap_check=cv2.VideoCapture(video_path);
            if not cap_check.isOpened(): raise FileNotFoundError(f"Cannot open video: {video_path}")
            original_fps=cap_check.get(cv2.CAP_PROP_FPS); total_frames=int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT)); cap_check.release()
            if original_fps <= 0: print(f"Warn: Invalid FPS ({original_fps}). Assuming 30."); original_fps = 30.0
            print(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename_base=f"output_{os.path.splitext(os.path.basename(video_path))[0]}_{ts}"
            output_path=os.path.join(VIDEO_OUTPUT_DIR, f"{output_filename_base}.mp4"); self.final_output_path = output_path

            # --- Start Threads ---
            print("Starting I/O threads...");
            self.reader_thread = threading.Thread(target=self._frame_reader_task, args=(video_path, original_fps), daemon=True)
            self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path,), daemon=True)
            self.reader_thread.start(); self.writer_thread.start()

            # --- Main Loop ---
            all_detections = []; frames_processed_count = 0
            frame_interval = max(1, round(original_fps / self.target_fps))
            total_target_frames = total_frames // frame_interval if frame_interval > 0 and total_frames > 0 else total_frames
            if self.perf: self.perf.start_processing(total_target_frames)
            frames_batch, frame_numbers_batch, frame_times_batch = [], [], []
            last_progress_print_time = time.monotonic(); print(f"Target FPS: {self.target_fps:.1f}, Processing every ~{frame_interval} frame(s).")
            while True:
                try:
                    frame_data = self.frame_read_queue.get(timeout=60)
                    if frame_data is None: self.frame_read_queue.task_done(); break
                    if self.perf: self.perf.start_timer('video_read')
                    frame, frame_number, frame_time_sec = frame_data
                    if self.perf: self.perf.end_timer('video_read')
                    if frame is None: self.frame_read_queue.task_done(); continue
                    frames_batch.append(frame); frame_numbers_batch.append(frame_number); frame_times_batch.append(frame_time_sec)

                    if len(frames_batch) >= self.batch_size:
                        batch_start_mono = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing'); input_batch = self._preprocess_batch(frames_batch); self.perf.end_timer('preprocessing')
                        if input_batch is None or input_batch.nelement() == 0:
                            debug_print("Skipping empty batch."); frames_batch.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()
                            for _ in range(len(frames_batch)): self.frame_read_queue.task_done() # Still mark as done
                            continue
                        # Inference
                        stream=None; idx=0;
                        if self.device=='cuda' and self.streams: stream=self.streams[self.current_stream_index%len(self.streams)]; idx=self.current_stream_index; self.current_stream_index+=1
                        with torch.cuda.stream(stream) if stream else torch.no_grad():
                           with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                              if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream); results=self.model(input_batch,verbose=False);
                              if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                           if stream: stream.synchronize()
                           if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                        # Post-process
                        batch_log, processed_frames = self._process_inference_results(results, frames_batch, frame_numbers_batch, frame_times_batch, start_datetime)
                        all_detections.extend(batch_log)
                        if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in results if hasattr(r,'boxes') and r.boxes))
                        # Write
                        if self.perf: self.perf.start_timer('video_write');
                        for pf in processed_frames:
                            if pf is not None: self.video_write_queue.put(pf)
                        if self.perf: self.perf.end_timer('video_write')
                        # Stats & Cleanup
                        batch_time = time.monotonic() - batch_start_mono; frames_processed_count += len(frames_batch)
                        if self.perf: self.perf.record_batch_processed(len(frames_batch), batch_time); self.perf.sample_system_metrics()
                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_f_time = frame_times_batch[-1] if frame_times_batch else 0
                            cleanup_tracking_data(self.vehicle_tracker, self.zone_tracker, last_f_time); cleanup_memory()
                            self.last_cleanup_time = current_mono_time
                            if self.perf: self.perf.end_timer('memory_cleanup')
                        # Progress
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0):
                             prog=self.perf.get_progress(); print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active:{self.vehicle_tracker.get_active_vehicle_count()}|ETA:{prog['eta']} ", end="")
                             last_progress_print_time = current_mono_time
                        # Clear Batch
                        frames_batch.clear(); frame_numbers_batch.clear(); frame_times_batch.clear(); del input_batch, results; cleanup_memory()
                    self.frame_read_queue.task_done()
                except queue.Empty: print("\nWarning: Frame reader queue timed out."); break # Exit if reader seems stuck
                except Exception as e: print(f"\nERROR during processing loop: {e}"); import traceback; traceback.print_exc(); self.stop_event.set(); break

            # --- Process Final Batch ---
            if frames_batch:
                print("\nProcessing final batch...")
                batch_start_mono=time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing'); input_batch=self._preprocess_batch(frames_batch); self.perf.end_timer('preprocessing')
                if input_batch is not None and input_batch.nelement() > 0:
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
                    for pf in processed_frames:
                        if pf is not None: self.video_write_queue.put(pf)
                    if self.perf: self.perf.end_timer('video_write')
                    batch_time=time.monotonic()-batch_start_mono; frames_processed_count+=len(frames_batch)
                    if self.perf: self.perf.record_batch_processed(len(frames_batch),batch_time)
                    del input_batch, results; cleanup_memory()
                else: debug_print("Skipping empty final batch.")
            print("\nVideo processing loop finished.")

            # --- Finalize & Report ---
            if self.vehicle_tracker and self.vehicle_tracker.active_vehicles:
                 print(f"Forcing exit for {len(self.vehicle_tracker.active_vehicles)} remaining...")
                 last_ts = start_datetime + timedelta(seconds=frame_times_batch[-1]) if frame_times_batch else start_datetime
                 last_fnum = frame_numbers_batch[-1] if frame_numbers_batch else (total_frames -1 if total_frames > 0 else 0)
                 force_logs = []
                 for v_id in list(self.vehicle_tracker.active_vehicles.keys()):
                     if self.vehicle_tracker.force_exit_vehicle(v_id, last_ts):
                         if self.perf: self.perf.record_vehicle_exit('forced_exit')
                         p_data = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id']==v_id and p['status']=='forced_exit'), None)
                         if p_data: date_str, time_str_ms = format_timestamp(last_ts); force_logs.append({'timestamp':time_str_ms, 'date':date_str, 'detection':f"{p_data['type']} FORCED (FROM {p_data.get('entry_direction','?').upper()})", 'frame_number':last_fnum, 'vehicle_id':v_id, 'status':'forced_exit', 'time_in_intersection':p_data.get('time_in_intersection','N/A')})
                         self.vehicle_tracker.remove_vehicle_data(v_id);
                         if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id)
                 all_detections.extend(force_logs)

            # --- Signal & Wait ---
            print("Signaling threads..."); self.stop_event.set() # Set stop event first
            if self.video_write_queue: self.video_write_queue.put(None) # Signal writer
            if self.frame_read_queue: # Try non-blocking put for reader
                try: self.frame_read_queue.put_nowait(None)
                except queue.Full: pass
            print("Waiting for threads..."); t_wait_start=time.monotonic()
            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30)
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=5)
            print(f"Threads joined in {time.monotonic()-t_wait_start:.2f}s")

            # --- Performance Summary & Reporting ---
            total_time_taken=0.0; final_fps=0.0; completed_paths_count_valid=0
            if self.perf:
                 self.perf.end_processing(); self.perf.print_summary()
                 total_time_taken=self.perf.total_time; final_fps=frames_processed_count/total_time_taken if total_time_taken>0 else 0
            final_completed_paths_list = []
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker:
                 final_completed_paths_list = self.vehicle_tracker.get_completed_paths()
                 for path in final_completed_paths_list: # Count valid exits
                     status=path.get('status'); entry_dir=path.get('entry_direction'); exit_dir=path.get('exit_direction')
                     is_valid_entry=entry_dir and entry_dir not in ['UNKNOWN',None]; is_valid_exit=exit_dir and exit_dir not in ['UNKNOWN','TIMEOUT','FORCED',None]
                     if status=='exited' and is_valid_entry and is_valid_exit: completed_paths_count_valid+=1

            # --- Pass zone_tracker to the report function ---
            excel_path = create_excel_report(detection_events=all_detections, completed_paths_data=final_completed_paths_list, start_datetime=start_datetime, zone_tracker=self.zone_tracker)
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)

            # --- Prepare result message ---
            actual_output_video = self.final_output_path if self.final_output_path else output_path
            result_msg=f"✅ Processing completed.\nOutput video: '{actual_output_video}'.\n"
            if excel_path: result_msg+=f"Excel report: {excel_path}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_msg+=f"Perf charts: 'performance_charts/'\n"
            result_msg+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_count}\nSTAT_TIME_SECONDS={total_time_taken:.2f}\nSTAT_FPS={final_fps:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid}\n--- End Stats ---"
            return result_msg

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e:
            print(f"\nFATAL ERROR during setup or processing: {e}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            # Return the specific error message to be displayed in Gradio or console
            return f"❌ Error: {e}"
        finally:
            # --- Final Cleanup ---
            try:
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
            except Exception as join_e: print(f"Error joining threads: {join_e}")
            self.model=None; self.vehicle_tracker=None; self.zone_tracker=None; self.perf=None; self.detection_zones_polygons=None
            self.frame_read_queue=None; self.video_write_queue=None; self.reader_thread=None; self.writer_thread=None; self.stop_event=None
            cleanup_memory()
            if self.processing_lock.locked():
                 try: self.processing_lock.release(); print("Processing finished, lock released.")
                 except RuntimeError as release_err: print(f"Warn: Error releasing lock: {release_err}")
            else: print("Processing finished (lock not held).")

# --- End of VideoProcessor class ---