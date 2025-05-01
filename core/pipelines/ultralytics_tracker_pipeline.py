# project/core/pipelines/ultralytics_tracker_pipeline.py
import cv2
import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict

# Import necessary components used *within* this pipeline
from config import (
    MEMORY_CLEANUP_INTERVAL, VEHICLE_TIMEOUTS, TRACK_HISTORY_LENGTH,
    CONF_THRESHOLD, IOU_THRESHOLD, MODEL_INPUT_SIZE
)
# Import cleanup timeout value
from tracking.cleanup import STALE_VEHICLE_TIMEOUT_SECONDS
# Use utils and overlay from the common locations
from core.utils import format_timestamp, is_valid_movement, cleanup_memory
from visualization import overlay as visual_overlay
# ZoneTracker is needed for type hinting if desired, but not strictly for functionality
# from tracking.zone_tracker import ZoneTracker

# Helper Function for Type Stabilization
def get_consistent_type_from_counts(type_counts):
    """Determines the most frequent type from a counts dictionary."""
    if not type_counts:
        return "Unknown"
    return max(type_counts, key=type_counts.get)

# --- Main Pipeline Function (Updated with Scaling) ---
def run_ultralytics_tracking(processor_instance, video_path, start_datetime, original_fps):
    """
    Processes video using model.track() with coordinate scaling,
    bottom-center tracking point, type stabilization, timeouts, cleanup, and trails.
    """
    # Get components from processor instance
    model = processor_instance.model
    zone_tracker = processor_instance.zone_tracker
    perf = processor_instance.perf
    video_write_queue = processor_instance.video_write_queue
    stop_event = processor_instance.stop_event
    final_output_path = processor_instance.final_output_path
    model_names = model.names
    # Get target frame dimensions from processor instance (from config)
    target_h, target_w = processor_instance.frame_shape[1], processor_instance.frame_shape[0]
    # Get tracker config file path from processor instance (from config)
    try:
         from config import TRACKER_CONFIG_FILE
         tracker_yaml_path = TRACKER_CONFIG_FILE
         print(f"Using Tracker Config: {tracker_yaml_path}")
    except ImportError:
         print("Warning: TRACKER_CONFIG_FILE not found in config. Using default.")
         # Decide default: 'bytetrack.yaml' might work if ultralytics finds it
         tracker_yaml_path = 'bytetrack.yaml'


    if not model or not zone_tracker:
         raise RuntimeError("Model or ZoneTracker not initialized in Processor Instance.")

    print(f"--- Running ULTRALYTICS_TRACKER Pipeline ---")

    total_frames = 0
    try:
        cap_check = cv2.VideoCapture(video_path)
        if cap_check.isOpened():
            total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video Info: Found approx {total_frames} total frames.")
        else:
            print("Warning: Could not open video to get total frame count for progress.")
        cap_check.release()
    except Exception as e:
        print(f"Warning: Error getting total frame count: {e}")

    # --- Stream Processing with model.track() ---
    results_generator = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker_yaml_path, # Use path from config
        conf=CONF_THRESHOLD,      # Use imported CONF_THRESHOLD
        iou=IOU_THRESHOLD,        # Use imported IOU_THRESHOLD
        imgsz=MODEL_INPUT_SIZE, # Optional: Add if you want to force internal size
        verbose=False
    )

    # --- State Management Dictionaries ---
    frame_number = 0
    track_history = {}          # {track_id: deque([(bc_x, bc_y), ...], maxlen=TRACK_HISTORY_LENGTH)}
    active_vehicles_state = {}  # {track_id: {'entry_direction':..., 'entry_time':..., 'status':..., 'type_counts':defaultdict(int), 'last_seen_time':float}}
    all_detections = []         # Log entries
    completed_paths = []        # Completed path dictionaries
    frames_processed_count = 0
    last_cleanup_time = time.monotonic()
    last_progress_print_time = time.monotonic()

    if perf: perf.start_processing(total_frames) # Start perf tracking

    # --- Main Frame Loop ---
    for results_per_frame in results_generator:
        if stop_event.is_set(): break

        original_frame = results_per_frame.orig_img
        if original_frame is None: continue
        # Ensure output frame matches target dimensions defined in config
        output_frame = cv2.resize(original_frame.copy(), (target_w, target_h))

        current_frame_time_sec = frame_number / original_fps if original_fps > 0 else frame_number / 30.0
        current_timestamp = start_datetime + timedelta(seconds=current_frame_time_sec)
        frame_local_detections = []
        processed_track_ids_this_frame = set()
        ids_to_remove_this_frame = set()

        batch_start_mono = time.monotonic()

        if results_per_frame.boxes is not None and results_per_frame.boxes.id is not None:
            # Get results - THESE ARE RELATIVE TO ORIGINAL FRAME SIZE
            boxes_xyxy_orig = results_per_frame.boxes.xyxy.cpu().numpy()
            track_ids = results_per_frame.boxes.id.int().cpu().tolist()
            classes = results_per_frame.boxes.cls.int().cpu().tolist()
            orig_h, orig_w = results_per_frame.orig_shape # Original frame dimensions

            # Calculate scaling factors once per frame
            w_scale = target_w / orig_w if orig_w > 0 else 1
            h_scale = target_h / orig_h if orig_h > 0 else 1

            for i, track_id in enumerate(track_ids):
                processed_track_ids_this_frame.add(track_id)
                current_bbox_orig = boxes_xyxy_orig[i]; cls_id = classes[i]
                current_detected_type = model_names.get(cls_id, f"CLS_{cls_id}")

                # --- Scale the current bounding box ---
                x1_orig, y1_orig, x2_orig, y2_orig = current_bbox_orig
                x1 = int(x1_orig * w_scale); y1 = int(y1_orig * h_scale)
                x2 = int(x2_orig * w_scale); y2 = int(y2_orig * h_scale)
                current_bbox_scaled = (x1, y1, x2, y2) # Box scaled to target size

                # Calculate SCALED bottom-center point
                bottom_center_point_scaled = ((x1 + x2) // 2, y2)

                # --- Update Track History (with SCALED Bottom-Center) ---
                if track_id not in track_history:
                     track_history[track_id] = deque(maxlen=TRACK_HISTORY_LENGTH)
                track_history[track_id].append(bottom_center_point_scaled)

                # --- Update Type Counts & Last Seen Time ---
                consistent_type = current_detected_type # Default
                if track_id in active_vehicles_state:
                    active_vehicles_state[track_id]['type_counts'][current_detected_type] += 1
                    active_vehicles_state[track_id]['last_seen_time'] = current_frame_time_sec
                    consistent_type = get_consistent_type_from_counts(active_vehicles_state[track_id]['type_counts'])
                # Else: state initialized on ENTRY event below

                # --- Zone Checking & Exit Logic (using SCALED points/boxes) ---
                prev_bottom_center_scaled = None
                if len(track_history.get(track_id, [])) > 1:
                     prev_bottom_center_scaled = track_history[track_id][-2]

                if prev_bottom_center_scaled is not None and zone_tracker.zones:
                    # Pass previous SCALED point and current SCALED BBOX
                    event_type, event_dir = zone_tracker.check_zone_transition(
                        prev_bottom_center_scaled, current_bbox_scaled, track_id, current_frame_time_sec
                    )
                    v_state = active_vehicles_state.get(track_id)

                    # Handle ENTRY/EXIT logic using consistent_type determined above
                    if event_type == "ENTRY":
                         if v_state and v_state.get('status')=='active':
                             stored_entry_dir = v_state.get('entry_direction')
                             if stored_entry_dir and stored_entry_dir != event_dir: # Inferred Exit
                                 active_zone_keys = list(zone_tracker.zones.keys()); stored_consistent_type = get_consistent_type_from_counts(v_state['type_counts'])
                                 if is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                     entry_time_stored = v_state.get('entry_time'); time_in_int = (current_timestamp - entry_time_stored).total_seconds() if entry_time_stored else None
                                     comp_path = {'id':track_id, 'type':stored_consistent_type, 'entry_direction':stored_entry_dir, 'exit_direction':event_dir, 'entry_time':entry_time_stored, 'exit_time':current_timestamp, 'status':'exited', 'time_in_intersection':round(time_in_int, 2) if time_in_int else None}
                                     completed_paths.append(comp_path); d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{stored_consistent_type} EXITED TO {event_dir.upper()} (FROM {stored_entry_dir.upper()}) [Inferred]", 'frame_number':frame_number, 'vehicle_id':track_id, 'status':'exit', 'time_in_intersection':time_in_int})
                                     if perf: perf.record_vehicle_exit('exited', time_in_int)
                                     ids_to_remove_this_frame.add(track_id)
                                 else: pass # Invalid move
                             else: # Re-entry/Same zone
                                 active_vehicles_state[track_id]['entry_direction'] = event_dir # Update entry dir? Or ignore? For now, update.
                                 active_vehicles_state[track_id]['entry_time'] = current_timestamp # Reset timer? Risky. Let's not reset time.
                                 active_vehicles_state[track_id]['last_seen_time'] = current_frame_time_sec
                                 active_vehicles_state[track_id]['type_counts'][current_detected_type] += 1 # Count type
                                 consistent_type = get_consistent_type_from_counts(active_vehicles_state[track_id]['type_counts']) # Re-check consistent type
                                 d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{consistent_type} ENTERED FROM {event_dir.upper()} [Re-entry]",'frame_number':frame_number,'vehicle_id':track_id,'status':'entry'})
                         else: # Standard new entry
                             active_vehicles_state[track_id] = {'entry_direction': event_dir, 'entry_time': current_timestamp, 'status': 'active', 'type_counts': defaultdict(int, {current_detected_type: 1}), 'last_seen_time': current_frame_time_sec}
                             consistent_type = current_detected_type
                             if perf: perf.record_vehicle_entry()
                             d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{consistent_type} ENTERED FROM {event_dir.upper()}",'frame_number':frame_number,'vehicle_id':track_id,'status':'entry'})

                    elif event_type == "EXIT": # Explicit Exit
                         if v_state and v_state.get('status') == 'active':
                             stored_entry_dir = v_state.get('entry_direction'); stored_consistent_type = get_consistent_type_from_counts(v_state['type_counts']); active_zone_keys = list(zone_tracker.zones.keys())
                             if stored_entry_dir and is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                 entry_time_stored = v_state.get('entry_time'); time_in_int = (current_timestamp - entry_time_stored).total_seconds() if entry_time_stored else None
                                 comp_path = {'id':track_id, 'type':stored_consistent_type, 'entry_direction':stored_entry_dir, 'exit_direction':event_dir, 'entry_time':entry_time_stored, 'exit_time':current_timestamp, 'status':'exited', 'time_in_intersection':round(time_in_int, 2) if time_in_int else None}
                                 completed_paths.append(comp_path); d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str,'date':d_str,'detection':f"{stored_consistent_type} EXITED TO {event_dir.upper()} (FROM {stored_entry_dir.upper()})",'frame_number':frame_number,'vehicle_id':track_id,'status':'exit','time_in_intersection':time_in_int})
                                 if perf: perf.record_vehicle_exit('exited', time_in_int)
                                 ids_to_remove_this_frame.add(track_id)


                # --- Visualization ---
                draw_status='detected'; entry_dir_draw=None; t_active_draw=None
                final_state = active_vehicles_state.get(track_id)
                # Determine display type based on latest state's counts
                display_type = get_consistent_type_from_counts(final_state['type_counts']) if final_state else consistent_type

                if final_state and final_state['status'] == 'active':
                     draw_status = 'active'; entry_dir_draw = final_state.get('entry_direction'); entry_time_draw = final_state.get('entry_time')
                     t_active_draw = (current_timestamp - entry_time_draw).total_seconds() if entry_time_draw else None

                # Draw using scaled coordinates and consistent type
                visual_overlay.draw_detection_box(output_frame, current_bbox_scaled, str(track_id), display_type, status=draw_status, entry_dir=entry_dir_draw, time_active=t_active_draw)

                # Draw Track Trail (using scaled history)
                trail_points = list(track_history.get(track_id, []))
                if len(trail_points) > 1:
                     visual_overlay.draw_tracking_trail(output_frame, trail_points, str(track_id))


            # --- End track processing loop ---

            # --- Cleanup completed tracks for this frame ---
            for tid_rem in ids_to_remove_this_frame:
                 active_vehicles_state.pop(tid_rem, None)
                 track_history.pop(tid_rem, None)

            # --- Timeout Check Logic ---
            if frame_number % int(original_fps if original_fps > 0 else 30) == 0:
                timeout_ids_to_remove = set()
                for t_id, state in list(active_vehicles_state.items()):
                    if state['status'] == 'active':
                        entry_t = state.get('entry_time')
                        v_type = get_consistent_type_from_counts(state.get('type_counts', {})) # Use consistent type
                        timeout_duration = VEHICLE_TIMEOUTS.get(v_type, VEHICLE_TIMEOUTS.get('default', 999))
                        if entry_t and (current_timestamp - entry_t).total_seconds() > timeout_duration:
                            time_in_int = (current_timestamp - entry_t).total_seconds()
                            comp_path = {'id':t_id, 'type':v_type, 'entry_direction':state.get('entry_direction', 'UNKNOWN'), 'exit_direction':'TIMEOUT', 'entry_time':entry_t, 'exit_time':current_timestamp, 'status':'timed_out', 'time_in_intersection':round(time_in_int, 2)}
                            completed_paths.append(comp_path); d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{v_type} TIMED OUT (FROM {state.get('entry_direction','UNKNOWN').upper()})", 'frame_number':frame_number, 'vehicle_id':t_id, 'status':'timeout', 'time_in_intersection':time_in_int})
                            if perf: perf.record_vehicle_exit('timed_out')
                            timeout_ids_to_remove.add(t_id)
                for tid_rem in timeout_ids_to_remove: active_vehicles_state.pop(tid_rem, None); track_history.pop(tid_rem, None)


        # --- Outside track processing (handle frames with NO tracks) ---
        else:
             # Still check for timeouts
             if frame_number % int(original_fps if original_fps > 0 else 30) == 0:
                 timeout_ids_to_remove = set()
                 for t_id, state in list(active_vehicles_state.items()):
                     if state['status'] == 'active':
                         entry_t = state.get('entry_time'); v_type = get_consistent_type_from_counts(state.get('type_counts', {})); timeout_duration = VEHICLE_TIMEOUTS.get(v_type, VEHICLE_TIMEOUTS.get('default', 999))
                         if entry_t and (current_timestamp - entry_t).total_seconds() > timeout_duration:
                             time_in_int = (current_timestamp - entry_t).total_seconds()
                             comp_path = {'id':t_id, 'type':v_type, 'entry_direction':state.get('entry_direction', 'UNKNOWN'), 'exit_direction':'TIMEOUT', 'entry_time':entry_t, 'exit_time':current_timestamp, 'status':'timed_out', 'time_in_intersection':round(time_in_int, 2)}
                             completed_paths.append(comp_path); d_str, t_str = format_timestamp(current_timestamp); frame_local_detections.append({'timestamp':t_str, 'date':d_str, 'detection':f"{v_type} TIMED OUT (FROM {state.get('entry_direction','UNKNOWN').upper()})", 'frame_number':frame_number, 'vehicle_id':t_id, 'status':'timeout', 'time_in_intersection':time_in_int})
                             if perf: perf.record_vehicle_exit('timed_out')
                             timeout_ids_to_remove.add(t_id)
                 for tid_rem in timeout_ids_to_remove: active_vehicles_state.pop(tid_rem, None); track_history.pop(tid_rem, None)

        # --- Log, Write Frame, Stats, Periodic Cleanup ---
        all_detections.extend(frame_local_detections)
        active_count = len(active_vehicles_state)
        # Update the call to add_status_overlay (already done, takes active_count)
        output_frame = visual_overlay.add_status_overlay(output_frame, frame_number, current_timestamp, active_count)
        if video_write_queue: video_write_queue.put(output_frame)

        frames_processed_count += 1
        batch_time = time.monotonic() - batch_start_mono
        if perf: perf.record_batch_processed(1, batch_time); perf.sample_system_metrics()
        current_mono_time = time.monotonic()

        # --- Periodic Cleanup (including stale state) ---
        if current_mono_time - last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
             if perf: perf.start_timer('memory_cleanup')
             zone_tracker.cleanup_cooldowns(current_frame_time_sec)
             stale_track_ids = set()
             if active_vehicles_state:
                for t_id, state in list(active_vehicles_state.items()):
                     last_seen = state.get('last_seen_time')
                     if last_seen and (current_frame_time_sec - last_seen > STALE_VEHICLE_TIMEOUT_SECONDS):
                         # print(f"STATE CLEANUP: Removing stale active track ID {t_id}")
                         stale_track_ids.add(t_id)
             for t_id_rem in stale_track_ids:
                 active_vehicles_state.pop(t_id_rem, None); track_history.pop(t_id_rem, None)
             cleanup_memory(); last_cleanup_time = current_mono_time
             if perf: perf.end_timer('memory_cleanup')

        # --- Progress Update ---
        if perf and (current_mono_time - last_progress_print_time > 1.0):
             prog = perf.get_progress(); print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active:{active_count}|ETA:{prog['eta']} ", end="")
             last_progress_print_time = current_mono_time

        frame_number += 1
        # --- End Frame Loop ---

    print("\nUltralytics tracker processing loop finished.")

    # --- Final Forced Exits ---
    if active_vehicles_state:
        print(f"Forcing exit for {len(active_vehicles_state)} remaining...")
        last_ts = current_timestamp; last_fnum = frame_number - 1
        force_logs = []
        for v_id, v_state in list(active_vehicles_state.items()):
            entry_time=v_state.get('entry_time'); time_in_int=(last_ts-entry_time).total_seconds() if entry_time else None
            # Use consistent type for final record
            c_type=get_consistent_type_from_counts(v_state.get('type_counts', {}))
            entry_dir=v_state.get('entry_direction','UNKNOWN')
            comp_path={'id':v_id, 'type':c_type, 'entry_direction':entry_dir, 'exit_direction':'FORCED', 'entry_time':entry_time, 'exit_time':last_ts, 'status':'forced_exit', 'time_in_intersection':round(time_in_int,2) if time_in_int else None}
            completed_paths.append(comp_path)
            if perf: perf.record_vehicle_exit('forced_exit')
            date_str, time_str_ms = format_timestamp(last_ts); force_logs.append({'timestamp':time_str_ms, 'date':date_str, 'detection':f"{c_type} FORCED (FROM {entry_dir.upper()})", 'frame_number':last_fnum, 'vehicle_id':v_id, 'status':'forced_exit', 'time_in_intersection':time_in_int})
        all_detections.extend(force_logs)


    return all_detections, completed_paths, frames_processed_count
# --- End run_ultralytics_tracking ---