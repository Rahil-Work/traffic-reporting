# project/core/pipelines/manual_batch_pipeline.py

import cv2
import numpy as np
import torch
import time
import queue
import threading
from datetime import datetime, timedelta
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Import necessary components used *within* this pipeline
from config import MEMORY_CLEANUP_INTERVAL, THREAD_COUNT, MIXED_PRECISION
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
from visualization import overlay as visual_overlay
from tracking.vehicle_tracker import VehicleTracker # Specific to this pipeline
from tracking.cleanup import cleanup_tracking_data   # Specific to this pipeline


# --- Helper: Frame Reader (Moved from VideoProcessor) ---
def _frame_reader_task(video_path, original_fps, target_fps, frame_read_queue, stop_event):
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print("ERROR: Reader failed to open video."); frame_read_queue.put(None); return
        frame_interval = max(1, round(original_fps / target_fps)); frame_count = 0; frames_yielded = 0
        while not stop_event.is_set():
            ret, frame = cap.read();
            if not ret: break
            if frame_count % frame_interval == 0:
                frame_time_seconds = frame_count / original_fps
                try:
                    # Simplified put logic assuming queue size is sufficient
                    frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                    frames_yielded += 1
                except queue.Full:
                    debug_print("Reader queue full, waiting briefly...")
                    time.sleep(0.1);
                    if stop_event.is_set(): break
                    try: frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0); frames_yielded += 1
                    except queue.Full: print("ERROR: Reader queue persistently full."); break
                except Exception as e: debug_print(f"Reader queue error: {e}"); break
            frame_count += 1
        debug_print(f"Reader finished. Read {frame_count}, yielded ~{frames_yielded}.")
    except Exception as e: print(f"ERROR in reader thread: {e}")
    finally:
        if cap: cap.release()
        if frame_read_queue:
            try: frame_read_queue.put(None, timeout=1.0)
            except queue.Full: pass


# --- Helper: Preprocess Batch (Moved from VideoProcessor) ---
@torch.no_grad()
def _preprocess_batch(frame_list, target_size, device):
    tensors = []
    for frame in frame_list:
        if frame is None or not isinstance(frame, np.ndarray): continue
        img=cv2.resize(frame,(target_size,target_size)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=img.transpose(2,0,1); img=np.ascontiguousarray(img); tensors.append(img)
    if not tensors: return None
    batch_tensor=torch.from_numpy(np.stack(tensors)).to(device)
    # Determine dtype based on global config (or pass MIXED_PRECISION?)
    dtype=torch.float16 if (MIXED_PRECISION and device=='cuda') else torch.float32
    batch_tensor = batch_tensor.to(dtype)/255.0; return batch_tensor


# --- Helper: Process Inference Results (Original Logic - Moved from VideoProcessor) ---
def _process_inference_results_manual(results, frames_batch, frame_numbers, frame_times, start_datetime, processor_instance):
    # Access shared components from processor_instance
    vehicle_tracker = processor_instance.vehicle_tracker
    zone_tracker = processor_instance.zone_tracker
    perf = processor_instance.perf
    tracker_lock = processor_instance.tracker_lock
    frame_shape = processor_instance.frame_shape
    model_input_size_config = processor_instance.model_input_size_config
    max_workers = processor_instance.max_workers
    model_names = processor_instance.model.names # Get class names from model

    batch_detections_list = [[] for _ in range(len(results))]
    processed_frames_list = [None] * len(results)

    def process_single_frame(idx):
        result=results[idx]; original_frame=frames_batch[idx]; frame_number=frame_numbers[idx]
        frame_time_sec=frame_times[idx]; current_timestamp=start_datetime+timedelta(seconds=frame_time_sec)
        output_frame=cv2.resize(original_frame.copy(), frame_shape); frame_local_detections=[]
        if not vehicle_tracker or not zone_tracker: print(f"ERR: Trackers missing frame {idx}"); return [], output_frame

        detections_this_frame = []; ids_processed_this_frame_map = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0]); v_type_name = model_names.get(cls_id, f"CLS_{cls_id}")
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                model_h,model_w = result.orig_shape if hasattr(result,'orig_shape') else (model_input_size_config, model_input_size_config)
                out_h,out_w = frame_shape[1], frame_shape[0]
                x1o=max(0,int(x1*out_w/model_w)); y1o=max(0,int(y1*out_h/model_h)); x2o=min(out_w-1,int(x2*out_w/model_w)); y2o=min(out_h-1,int(y2*out_h/model_h))
                center_pt = ((x1o+x2o)//2, (y1o+y2o)//2)
                detections_this_frame.append({'box_coords': (x1o,y1o,x2o,y2o), 'center': center_pt, 'type': v_type_name})

        ids_seen_in_lock = set(); ids_to_remove_this_frame = set()
        with tracker_lock: # --- Start Lock ---
            if perf: perf.start_timer('tracking_update')
            for det_info in detections_this_frame: # Match detections
                center_point, v_type_name = det_info['center'], det_info['type']
                matched_id = vehicle_tracker.find_best_match(center_point, frame_time_sec)
                v_id, c_type = vehicle_tracker.update_track(matched_id if matched_id else vehicle_tracker._get_next_id(), center_point, frame_time_sec, v_type_name)
                ids_processed_this_frame_map[v_id] = {'center':center_point, 'box':det_info['box_coords'], 'type':c_type, 'event':None, 'status':'detected'}
                ids_seen_in_lock.add(v_id)

            for v_id in list(ids_processed_this_frame_map.keys()): # Check zones
                center_point, c_type = ids_processed_this_frame_map[v_id]['center'], ids_processed_this_frame_map[v_id]['type']
                current_bbox = ids_processed_this_frame_map[v_id]['box'] # Get box for bottom-center check
                prev_pos=None
                if v_id in vehicle_tracker.tracking_history and len(vehicle_tracker.tracking_history[v_id])>1: prev_pos=vehicle_tracker.tracking_history[v_id][-2]

                if prev_pos and zone_tracker and zone_tracker.zones:
                    if perf: perf.start_timer('zone_checking')
                    # Use the check that takes bbox for current point
                    event_type, event_dir = zone_tracker.check_zone_transition(prev_pos, current_bbox, v_id, frame_time_sec)
                    if perf: perf.end_timer('zone_checking')

                    # --- Use Inferred Exit Logic (adopted from ultralytics pipeline) ---
                    if event_type == "ENTRY":
                        v_state = vehicle_tracker.active_vehicles.get(v_id)
                        if v_state and v_state.get('status')=='active':
                            stored_entry_dir = v_state.get('entry_direction')
                            if stored_entry_dir and stored_entry_dir != event_dir:
                                # Inferred Exit
                                active_zone_keys = list(zone_tracker.zones.keys())
                                if is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                    entry_time_stored = v_state.get('entry_time')
                                    time_in_int = (current_timestamp - entry_time_stored).total_seconds() if entry_time_stored else None
                                    # Manually add completion record
                                    comp_path = {'id': v_id, 'type': v_state.get('type', c_type), 'entry_direction': stored_entry_dir, 'exit_direction': event_dir, 'entry_time': entry_time_stored, 'exit_time': current_timestamp, 'status': 'exited', 'time_in_intersection': round(time_in_int, 2) if time_in_int else None}
                                    vehicle_tracker.completed_paths.append(comp_path) # Add directly
                                    # Log event
                                    d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{v_state.get('type', c_type)} EXITED TO {event_dir.upper()} (FROM {stored_entry_dir.upper()}) [Inferred]", 'frame_number':frame_number, 'vehicle_id':v_id, 'status':'exit', 'time_in_intersection':time_in_int})
                                    if perf: perf.record_vehicle_exit('exited', time_in_int)
                                    ids_to_remove_this_frame.add(v_id) # Mark for removal
                                    ids_processed_this_frame_map[v_id]['event'] = 'EXIT'
                                else: pass # Invalid inferred move
                            else: # Re-entry/Same zone -> standard entry
                                if vehicle_tracker.register_entry(v_id, event_dir, current_timestamp, center_point, c_type):
                                    if perf: perf.record_vehicle_entry(); d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{c_type} ENTERED FROM {event_dir.upper()}",'frame_number':frame_number,'vehicle_id':v_id,'status':'entry'})
                                    ids_processed_this_frame_map[v_id]['event'] = 'ENTRY'
                        else: # Standard new entry
                            if vehicle_tracker.register_entry(v_id, event_dir, current_timestamp, center_point, c_type):
                                if perf: perf.record_vehicle_entry(); d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{c_type} ENTERED FROM {event_dir.upper()}",'frame_number':frame_number,'vehicle_id':v_id,'status':'entry'})
                                ids_processed_this_frame_map[v_id]['event'] = 'ENTRY'

                    elif event_type == "EXIT": # Explicit Exit
                        v_state = vehicle_tracker.active_vehicles.get(v_id)
                        if v_state and v_state.get('status') == 'active':
                            entry_dir_trk = v_state.get('entry_direction')
                            active_zone_keys = list(zone_tracker.zones.keys())
                            if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir, active_zone_keys):
                                success,t_in_int=vehicle_tracker.register_exit(v_id,event_dir,current_timestamp,center_point)
                                if success:
                                    if perf:perf.record_vehicle_exit('exited',t_in_int); d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{c_type} EXITED TO {event_dir.upper()} (FROM {entry_dir_trk.upper()})",'frame_number':frame_number,'vehicle_id':v_id,'status':'exit','time_in_intersection':t_in_int})
                                    ids_processed_this_frame_map[v_id]['event']='EXIT'; ids_to_remove_this_frame.add(v_id)

            # Check timeouts using VehicleTracker method
            timed_out_ids=vehicle_tracker.check_timeouts(current_timestamp)
            for v_id_to in timed_out_ids:
                ids_to_remove_this_frame.add(v_id_to);
                if perf:perf.record_vehicle_exit('timed_out')
                p_data=next((p for p in reversed(vehicle_tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'),None)
                if p_data: d_str,t_str=format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{p_data['type']} TIMED OUT (FROM {p_data.get('entry_direction','UNKNOWN').upper()})",'frame_number':frame_number,'vehicle_id':v_id_to,'status':'timeout','time_in_intersection':p_data.get('time_in_intersection','N/A')})
                if v_id_to in ids_processed_this_frame_map: ids_processed_this_frame_map[v_id_to]['status']='timed_out'

            vehicle_tracker.increment_misses(ids_seen_in_lock) # Call miss counter

            for v_id_rem in ids_to_remove_this_frame: # Remove exited/timed_out
                vehicle_tracker.remove_vehicle_data(v_id_rem)
                if zone_tracker: zone_tracker.remove_vehicle_data(v_id_rem)
            if perf: perf.end_timer('tracking_update')
        # --- End Lock ---

        # --- Drawing Pass ---
        if perf: perf.start_timer('drawing')
        if zone_tracker and zone_tracker.zones: output_frame = visual_overlay.draw_zones(output_frame, zone_tracker)
        active_vehicle_snapshot = vehicle_tracker.active_vehicles.copy()
        for v_id_draw, data in ids_processed_this_frame_map.items():
            center_pt_draw, box_draw, c_type_draw = data['center'], data['box'], data['type']
            current_draw_status = data.get('status', 'detected'); display_status = "detected"; entry_dir_draw = None; t_active_draw = None
            if v_id_draw in active_vehicle_snapshot:
                 v_state = active_vehicle_snapshot[v_id_draw]
                 if v_state.get('status') == 'active': display_status='active'; entry_dir_draw=v_state.get('entry_direction'); t_active_draw=(current_timestamp - v_state['entry_time']).total_seconds() if v_state.get('entry_time') else None
            if current_draw_status != 'timed_out': visual_overlay.draw_detection_box(output_frame, box_draw, v_id_draw, c_type_draw, status=display_status, entry_dir=entry_dir_draw, time_active=t_active_draw)
            trail = vehicle_tracker.get_tracking_trail(v_id_draw); visual_overlay.draw_tracking_trail(output_frame, trail, v_id_draw)
            event_marker = data.get('event');
            if event_marker: visual_overlay.draw_event_marker(output_frame, center_pt_draw, event_marker, v_id_draw[-4:])
        current_active_count = vehicle_tracker.get_active_vehicle_count()
        output_frame = visual_overlay.add_status_overlay(output_frame, frame_number, current_timestamp, current_active_count, vehicle_tracker)
        if perf: perf.end_timer('drawing')
        return frame_local_detections, output_frame
        # --- End process_single_frame ---

    # --- Execute in parallel ---
    if perf: perf.start_timer('detection_processing')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
         futures = {executor.submit(process_single_frame, i): i for i in range(len(results))}
         for future in concurrent.futures.as_completed(futures):
             idx = futures[future]
             try: frame_detections, processed_frame = future.result(); batch_detections_list[idx] = frame_detections; processed_frames_list[idx] = processed_frame
             except Exception as exc: print(f'Frame {idx} failed: {exc}'); import traceback; traceback.print_exc(); processed_frames_list[idx] = cv2.resize(frames_batch[idx].copy(), frame_shape) # Fallback
    final_batch_detections = [det for frame_list in batch_detections_list for det in frame_list]
    final_processed_frames = processed_frames_list
    if perf: perf.end_timer('detection_processing')
    return final_batch_detections, final_processed_frames
# --- End _process_inference_results_manual ---


# --- Main Pipeline Function ---
def run_manual_batch(processor_instance, video_path, start_datetime, original_fps):
    """
    Processes video using manual batching and the custom VehicleTracker.
    Args:
        processor_instance: The instance of VideoProcessor.
        video_path (str): Path to the input video.
        start_datetime (datetime): Starting timestamp.
        original_fps (float): Original FPS of the video.
    Returns:
        tuple: (all_detections_log, completed_paths_list, frames_processed_count)
    """
    # Get components from the processor instance
    model = processor_instance.model
    zone_tracker = processor_instance.zone_tracker
    perf = processor_instance.perf
    device = processor_instance.device
    target_fps = processor_instance.target_fps
    batch_size = processor_instance.batch_size
    model_input_size_config = processor_instance.model_input_size_config

    # Create specific instances needed only for this pipeline
    vehicle_tracker = VehicleTracker()
    processor_instance.vehicle_tracker = vehicle_tracker # Allow status overlay access

    stop_event = processor_instance.stop_event # Use the shared stop event
    frame_read_queue = queue.Queue(maxsize=batch_size * 4)
    video_write_queue = processor_instance.video_write_queue # Use shared writer queue

    reader_thread = threading.Thread(target=_frame_reader_task, args=(video_path, original_fps, target_fps, frame_read_queue, stop_event), daemon=True)
    reader_thread.start()

    print(f"--- Running MANUAL_BATCH Pipeline ---")

    all_detections = []; frames_processed_count = 0
    # total_frames estimate might be passed or recalculated if needed for perf tracker
    # total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) # Example
    # if perf: perf.start_processing(total_frames / max(1, round(original_fps / target_fps))) # Adjust total frame estimate
    if perf: perf.start_processing(0) # Start perf tracking (total frames unknown here)

    frames_batch, frame_numbers_batch, frame_times_batch = [], [], []
    last_progress_print_time = time.monotonic()
    last_cleanup_time = time.monotonic()

    while True:
        try:
            frame_data = frame_read_queue.get(timeout=60)
            if frame_data is None: frame_read_queue.task_done(); break
            if perf: perf.start_timer('video_read')
            frame, frame_number, frame_time_sec = frame_data
            if perf: perf.end_timer('video_read')
            if frame is None: frame_read_queue.task_done(); continue

            frames_batch.append(frame)
            frame_numbers_batch.append(frame_number)
            frame_times_batch.append(frame_time_sec)

            if len(frames_batch) >= batch_size:
                batch_start_mono = time.monotonic()
                if perf: perf.start_timer('preprocessing'); input_batch = _preprocess_batch(frames_batch, model_input_size_config, device); perf.end_timer('preprocessing')
                if input_batch is None or input_batch.nelement() == 0:
                    debug_print("Skipping empty batch."); frames_batch.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()
                    for _ in range(len(frames_batch)): frame_read_queue.task_done()
                    continue

                # --- Inference ---
                # (Using model directly, no streams assumed here unless passed in processor_instance)
                if perf and hasattr(perf, 'start_event'): perf.start_event.record() # Record on default stream
                with torch.amp.autocast(device_type=device, enabled=MIXED_PRECISION and device=='cuda'):
                    results = model(input_batch, verbose=False)
                if perf and hasattr(perf, 'end_event'): perf.end_event.record(); torch.cuda.synchronize() # Sync default stream
                if perf and hasattr(perf, 'start_event'): perf.record_inference_time_gpu(perf.start_event, perf.end_event)


                # --- Post-processing (Uses helper function) ---
                batch_log, processed_frames = _process_inference_results_manual(
                    results, frames_batch, frame_numbers_batch, frame_times_batch, start_datetime, processor_instance
                )
                all_detections.extend(batch_log)
                if perf: perf.record_detection(sum(len(r.boxes) for r in results if hasattr(r,'boxes') and r.boxes))

                # --- Write Output ---
                if perf: perf.start_timer('video_write');
                for pf in processed_frames:
                    if pf is not None: video_write_queue.put(pf)
                if perf: perf.end_timer('video_write')

                # --- Stats & Cleanup ---
                batch_time = time.monotonic() - batch_start_mono; frames_processed_count += len(frames_batch)
                perf = processor_instance.perf if hasattr(processor_instance, 'perf') else None
                if perf: perf.record_batch_processed(len(frames_batch), batch_time); perf.sample_system_metrics()
                current_mono_time = time.monotonic()
                if current_mono_time - last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                    if perf: perf.start_timer('memory_cleanup')
                    last_f_time = frame_times_batch[-1] if frame_times_batch else 0
                    # Pass the locally created vehicle_tracker to cleanup
                    cleanup_tracking_data(vehicle_tracker, zone_tracker, last_f_time); cleanup_memory()
                    last_cleanup_time = current_mono_time
                    if perf: perf.end_timer('memory_cleanup')

                # --- Progress Update ---
                if perf and (current_mono_time - last_progress_print_time > 1.0):
                     prog = perf.get_progress(); print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active:{vehicle_tracker.get_active_vehicle_count()}|ETA:{prog['eta']} ", end="")
                     last_progress_print_time = current_mono_time

                # --- Clear Batch ---
                frames_batch.clear(); frame_numbers_batch.clear(); frame_times_batch.clear(); del input_batch, results; cleanup_memory()

            frame_read_queue.task_done()
        except queue.Empty: print("\nWarning: Frame reader queue timed out."); break
        except Exception as e: print(f"\nERROR during manual pipeline loop: {e}"); import traceback; traceback.print_exc(); stop_event.set(); break

    # --- Process Final Batch ---
    if frames_batch:
        print("\nProcessing final manual batch...")
        batch_start_mono=time.monotonic()
        if perf: perf.start_timer('preprocessing'); input_batch=_preprocess_batch(frames_batch, model_input_size_config, device); perf.end_timer('preprocessing')
        if input_batch is not None and input_batch.nelement() > 0:
             if perf and hasattr(perf, 'start_event'): perf.start_event.record()
             with torch.amp.autocast(device_type=device,enabled=MIXED_PRECISION and device=='cuda'):
                 results = model(input_batch,verbose=False)
             if perf and hasattr(perf, 'end_event'): perf.end_event.record(); torch.cuda.synchronize()
             if perf and hasattr(perf, 'start_event'): perf.record_inference_time_gpu(perf.start_event, perf.end_event)

             batch_log, processed_frames = _process_inference_results_manual(results, frames_batch, frame_numbers_batch, frame_times_batch, start_datetime, processor_instance)
             all_detections.extend(batch_log)
             if perf: perf.start_timer('video_write');
             for pf in processed_frames:
                 if pf is not None: video_write_queue.put(pf)
             if perf: perf.end_timer('video_write')
             batch_time=time.monotonic()-batch_start_mono; frames_processed_count+=len(frames_batch)
             if perf: perf.record_batch_processed(len(frames_batch),batch_time)
             del input_batch, results; cleanup_memory()
        else: debug_print("Skipping empty final batch.")
    print("\nManual batch processing loop finished.")

    # --- Final Forced Exits (Using local vehicle_tracker) ---
    completed_paths_list = []
    if vehicle_tracker and vehicle_tracker.active_vehicles:
         print(f"Forcing exit for {len(vehicle_tracker.active_vehicles)} remaining...")
         last_ts = start_datetime + timedelta(seconds=frame_times_batch[-1]) if frame_times_batch else start_datetime
         last_fnum = frame_numbers_batch[-1] if frame_numbers_batch else 0 # Estimate last frame number
         force_logs = []
         for v_id in list(vehicle_tracker.active_vehicles.keys()):
             if vehicle_tracker.force_exit_vehicle(v_id, last_ts):
                 if perf: perf.record_vehicle_exit('forced_exit')
                 p_data = next((p for p in reversed(vehicle_tracker.completed_paths) if p['id']==v_id and p['status']=='forced_exit'),None)
                 if p_data: date_str, time_str_ms = format_timestamp(last_ts); force_logs.append({'timestamp':time_str_ms, 'date':date_str, 'detection':f"{p_data['type']} FORCED (FROM {p_data.get('entry_direction','?').upper()})", 'frame_number':last_fnum, 'vehicle_id':v_id, 'status':'forced_exit', 'time_in_intersection':p_data.get('time_in_intersection','N/A')})
                 # No need to explicitly remove here as vehicle_tracker instance is local to this function
         all_detections.extend(force_logs)
         # Get final completed paths from the local tracker
         completed_paths_list = vehicle_tracker.get_completed_paths()
    elif vehicle_tracker:
         # Still get paths even if no active vehicles remained
         completed_paths_list = vehicle_tracker.get_completed_paths()


    # Signal reader thread (already done by loop exit/error) and wait
    if reader_thread and reader_thread.is_alive():
        print("Waiting for reader thread to finish...")
        reader_thread.join(timeout=5)

    processor_instance.vehicle_tracker = None # Clear reference

    return all_detections, completed_paths_list, frames_processed_count
# --- End run_manual_batch ---