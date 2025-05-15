# project/tracking/vehicle_tracker.py
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from config import (
    REIDENTIFICATION_TIMEOUT, VEHICLE_TIMEOUTS,
    TRACK_HISTORY_LENGTH,
    MAX_MATCHING_DISTANCE,
    MAX_CONSECUTIVE_MISSES
)
from core.utils import debug_print

class VehicleTracker:
    """Manages vehicle tracking state and logic."""
    def __init__(self):
        self.next_track_id = 0
        self.active_vehicles = {}   # {id: {data}}
        self.last_positions = {}    # {id: (x, y)}
        self.last_seen_time = {}    # {id: frame_time_seconds}
        self.tracking_history = {}  # {id: deque([(x,y), ...])}
        self.vehicle_type_history = {}  # {id: {type_name: count}}
        self.completed_paths = []   # Stores data of exited/timed_out vehicles
        self.consecutive_misses = {}    # {id: count}
        self.completed_vehicle_ids = set()  # Store IDs that have finished their journey
        self.max_reid_timeout = REIDENTIFICATION_TIMEOUT
        self.max_misses = MAX_CONSECUTIVE_MISSES

    def _get_next_id(self): self.next_track_id += 1; return f"v_{self.next_track_id}"

    def find_best_match(self, center_point, current_frame_time):
        """Finds the best match, prioritizing non-stale tracks."""
        min_dist_sq = (MAX_MATCHING_DISTANCE * MAX_MATCHING_DISTANCE) # Standard max distance squared

        best_non_stale_id = None
        min_non_stale_dist_sq = min_dist_sq

        best_stale_id = None
        min_stale_dist_sq = min_dist_sq # Use same initial limit for stale, but only consider if no non-stale found

        # Iterate only over IDs that are NOT in completed_vehicle_ids
        candidate_ids = [
            v_id for v_id in self.last_positions.keys()
            if v_id not in self.completed_vehicle_ids
        ]

        for v_id in candidate_ids:
            last_pos = self.last_positions.get(v_id)
            if not last_pos: continue

            last_seen = self.last_seen_time.get(v_id, -1)
            # --- Basic Time Check First ---
            if current_frame_time - last_seen > self.max_reid_timeout:
                  continue # Definitely too old

            miss_count = self.consecutive_misses.get(v_id, 0)
            is_stale_due_to_misses = miss_count > self.max_misses

            dx = center_point[0]-last_pos[0]; dy = center_point[1]-last_pos[1]; dist_sq = dx*dx + dy*dy

            # --- Check against standard distance limit ---
            if dist_sq < min_dist_sq:
                if not is_stale_due_to_misses:
                    # Found a non-stale match within range
                    if dist_sq < min_non_stale_dist_sq:
                         min_non_stale_dist_sq = dist_sq
                         best_non_stale_id = v_id
                else:
                    # Found a stale match within range
                    if dist_sq < min_stale_dist_sq:
                         min_stale_dist_sq = dist_sq
                         best_stale_id = v_id

        # --- Prioritize non-stale match ---
        if best_non_stale_id is not None:
            debug_print(f"Match: Found non-stale V:{best_non_stale_id} (Dist:{min_non_stale_dist_sq**0.5:.1f})")
            return best_non_stale_id
        elif best_stale_id is not None:
            # Only return stale match if no non-stale one was found within range
            debug_print(f"Match: Found STALE V:{best_stale_id} (Dist:{min_stale_dist_sq**0.5:.1f}) - No non-stale match.")
            return best_stale_id
        else:
            # No match found within distance/time limits
            return None

    def get_consistent_type(self, vehicle_id, current_type_name):
        if vehicle_id not in self.vehicle_type_history: self.vehicle_type_history[vehicle_id] = {current_type_name: 1}; return current_type_name
        counts = self.vehicle_type_history[vehicle_id]; counts[current_type_name] = counts.get(current_type_name, 0) + 1
        return max(counts, key=counts.get)

    def update_track(self, vehicle_id, center_point, current_frame_time, vehicle_type_name):
        # Ensure this ID is not trying to be updated if it's already completed
        if vehicle_id in self.completed_vehicle_ids:
            debug_print(f"WARN: Attempted to update_track for completed V:{vehicle_id}. Ignoring.")
            return None, None # Indicate failure or no update

        self.last_positions[vehicle_id] = center_point
        self.last_seen_time[vehicle_id] = current_frame_time
        consistent_type = self.get_consistent_type(vehicle_id, vehicle_type_name)
        if vehicle_id not in self.tracking_history:
            self.tracking_history[vehicle_id] = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.tracking_history[vehicle_id].append(center_point)
        self.consecutive_misses[vehicle_id] = 0
        return vehicle_id, consistent_type
    
    def increment_misses(self, ids_seen_this_frame):
        missed_ids = []
        for v_id in list(self.last_positions.keys()):
            if v_id not in ids_seen_this_frame and v_id not in self.completed_vehicle_ids: # <<< MODIFIED
                self.consecutive_misses[v_id] = self.consecutive_misses.get(v_id, 0) + 1
                missed_ids.append(v_id)
                if self.consecutive_misses[v_id] > self.max_misses // 2:
                    debug_print(f"WARN: V:{v_id} miss count high: {self.consecutive_misses[v_id]}")
        return missed_ids

    def register_entry(self, vehicle_id, entry_direction, entry_time, entry_point, vehicle_type):
        # An ID should only be registered if it's entirely new to the active system OR if this is its first ever registration.
        # completed_vehicle_ids ensures a "spent" ID cannot re-enter.
        if vehicle_id in self.completed_vehicle_ids:
            debug_print(f"WARN: V:{vehicle_id} has already completed a journey. Ignoring re-entry attempt for this ID.")
            return False
        if vehicle_id not in self.active_vehicles: # Standard new entry
            self.active_vehicles[vehicle_id] = {
                'id': vehicle_id, 'type': vehicle_type, 'entry_direction': entry_direction,
                'entry_time': entry_time, 'entry_point': entry_point,
                'exit_direction': None, 'exit_time': None, 'exit_point': None, 'status': 'active'
            }
            # Initialize other tracking states for this new ID
            self.last_positions[vehicle_id] = entry_point
            self.last_seen_time[vehicle_id] = (entry_time - datetime.min.replace(tzinfo=entry_time.tzinfo)).total_seconds() # Needs careful handling if entry_time is not video start
                                                                        # This assumes entry_time is a datetime obj. If it's frame_time_sec, use that directly.
                                                                        # This will be updated by update_track anyway.
            self.tracking_history[vehicle_id] = deque([entry_point], maxlen=TRACK_HISTORY_LENGTH)
            self.vehicle_type_history[vehicle_id] = {vehicle_type: 1}
            self.consecutive_misses[vehicle_id] = 0
            debug_print(f"ENTRY: V:{vehicle_id} ({vehicle_type}) from {entry_direction}")
            return True
        
        # This case should ideally not be hit often if find_best_match is working and ID is not completed.
        # It means we are trying to register_entry for an ID that's already in active_vehicles but not completed.
        debug_print(f"WARN: V:{vehicle_id} already active. Entry from {entry_direction} ignored for already active vehicle.")
        return False

    def _finalize_and_complete_path(self, vehicle_id, data_to_complete):
        """Helper to add to completed_paths and mark ID as spent."""
        self.completed_paths.append(data_to_complete.copy())
        self.completed_vehicle_ids.add(vehicle_id)
        self.remove_vehicle_data(vehicle_id) # Clean up active tracking data for this ID

    def register_exit(self, vehicle_id, exit_direction, exit_time, exit_point):
        if vehicle_id in self.active_vehicles and self.active_vehicles[vehicle_id]['status'] == 'active':
            data = self.active_vehicles[vehicle_id]
            final_type = data.get('type', 'UnknownType')
            if vehicle_id in self.vehicle_type_history and self.vehicle_type_history[vehicle_id]:
                counts = self.vehicle_type_history[vehicle_id]
                final_type = max(counts, key=counts.get)
            data['type'] = final_type

            data.update({
                'exit_direction': exit_direction, 'exit_time': exit_time,
                'exit_point': exit_point, 'status': 'exited'
            })
            time_in_int = (exit_time - data['entry_time']).total_seconds()
            data['time_in_intersection'] = round(time_in_int, 2)
            
            self._finalize_and_complete_path(vehicle_id, data) # <<< USE HELPER
            debug_print(f"EXIT: V:{vehicle_id} ({final_type}) to {exit_direction}. Time: {time_in_int:.2f}s. ID retired.")
            return True, time_in_int
        
        debug_print(f"WARN: Exit ignored for inactive/already exited V:{vehicle_id}")
        return False, None

    def check_timeouts(self, current_time: datetime):
        timed_out_ids_this_pass = []
        for v_id, data in list(self.active_vehicles.items()): # Iterate copy as we modify
            if data['status'] == 'active': # Only consider active vehicles
                if v_id in self.completed_vehicle_ids: # Should not happen if logic is correct
                    debug_print(f"ERROR: V:{v_id} is in active_vehicles but also in completed_vehicle_ids during timeout check.")
                    self.remove_vehicle_data(v_id) # Clean up inconsistent state
                    continue

                time_in_int = (current_time - data['entry_time']).total_seconds()
                v_type_current = data.get('type', 'default')
                timeout_duration = VEHICLE_TIMEOUTS.get(v_type_current, VEHICLE_TIMEOUTS['default'])

                if time_in_int > timeout_duration:
                    final_type = data.get('type', 'UnknownType')
                    if v_id in self.vehicle_type_history and self.vehicle_type_history[v_id]:
                        final_type = max(self.vehicle_type_history[v_id], key=self.vehicle_type_history[v_id].get)
                    data['type'] = final_type

                    miss_count = self.consecutive_misses.get(v_id, 0) # Get current miss count
                    debug_print(f"TIMEOUT: V:{v_id} ({final_type}) after {time_in_int:.1f}s. Missed frames: {miss_count}. ID retired.")
                    data.update({
                        'status': 'timed_out', 'exit_time': current_time,
                        'exit_direction': 'TIMEOUT', 'time_in_intersection': round(time_in_int, 2)
                    })
                    self._finalize_and_complete_path(v_id, data)
                    timed_out_ids_this_pass.append(v_id)
        return timed_out_ids_this_pass

    def force_exit_vehicle(self, vehicle_id, exit_time: datetime):
        if vehicle_id in self.active_vehicles and self.active_vehicles[vehicle_id]['status'] == 'active':
            if vehicle_id in self.completed_vehicle_ids: # Defensive check
                debug_print(f"ERROR: V:{vehicle_id} is in active_vehicles but also completed during force_exit.")
                self.remove_vehicle_data(vehicle_id)
                return False

            data = self.active_vehicles[vehicle_id]
            final_type = data.get('type', 'UnknownType')
            if vehicle_id in self.vehicle_type_history and self.vehicle_type_history[vehicle_id]:
                final_type = max(self.vehicle_type_history[vehicle_id], key=self.vehicle_type_history[vehicle_id].get)
            data['type'] = final_type
            time_in_int = (exit_time - data['entry_time']).total_seconds()
            data.update({
                'status': 'forced_exit', 'exit_time': exit_time,
                'exit_direction': 'FORCED', 'time_in_intersection': round(time_in_int, 2)
            })
            self._finalize_and_complete_path(vehicle_id, data)
            debug_print(f"FORCED EXIT: V:{vehicle_id} ({final_type}). ID retired.")
            return True
        return False

    def remove_vehicle_data(self, vehicle_id):
        # This function now primarily cleans up tracking dictionaries.
        # active_vehicles.pop() is handled by the _finalize_and_complete_path or other callers.
        removed_count = 0
        if self.active_vehicles.pop(vehicle_id, None) is not None: removed_count +=1
        if self.last_positions.pop(vehicle_id, None) is not None: removed_count +=1
        if self.last_seen_time.pop(vehicle_id, None) is not None: removed_count +=1
        if self.tracking_history.pop(vehicle_id, None) is not None: removed_count +=1
        if self.vehicle_type_history.pop(vehicle_id, None) is not None: removed_count +=1
        if self.consecutive_misses.pop(vehicle_id, None) is not None: removed_count +=1
        
        # Do NOT remove from self.completed_vehicle_ids here as this function is called
        # when an ID is added to completed_vehicle_ids.
        if removed_count > 0:
            debug_print(f"Cleared tracking data for V:{vehicle_id} (now completed or removed).")
        return removed_count > 0

    def get_active_vehicle_count(self):
        # Count active vehicles that are not yet completed
        return len([v_id for v_id in self.active_vehicles if v_id not in self.completed_vehicle_ids and self.active_vehicles[v_id]['status'] == 'active'])

    def get_completed_paths(self):
        return self.completed_paths

    def get_tracking_trail(self, vehicle_id):
        return list(self.tracking_history.get(vehicle_id, []))