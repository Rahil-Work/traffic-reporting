# project/tracking/vehicle_tracker.py
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from config import (
    REIDENTIFICATION_TIMEOUT, VEHICLE_TIMEOUTS,
    TRACK_HISTORY_LENGTH,
    # MAX_MATCHING_DISTANCE, # <-- REMOVED IMPORT
    MAX_CONSECUTIVE_MISSES
)
from core.utils import debug_print

class VehicleTracker:
    """Manages vehicle tracking state and logic."""
    def __init__(self):
        self.next_track_id = 0
        self.active_vehicles = {} # {id: {data}}
        self.last_positions = {} # {id: (x, y)}
        self.last_seen_time = {} # {id: frame_time_seconds}
        self.tracking_history = {} # {id: deque([(x,y), ...])}
        self.vehicle_type_history = {} # {id: {type_name: count}}
        self.completed_paths = [] # Stores data of exited/timed_out vehicles
        # --- Add state for consecutive misses ---
        self.consecutive_misses = {} # {id: count}
        # --- Store config values ---
        self.max_reid_timeout = REIDENTIFICATION_TIMEOUT
        # self.max_match_dist_sq = MAX_MATCHING_DISTANCE * MAX_MATCHING_DISTANCE # <-- REMOVED
        self.max_misses = MAX_CONSECUTIVE_MISSES

    def _get_next_id(self): self.next_track_id += 1; return f"v_{self.next_track_id}"

    def find_best_match(self, center_point, current_frame_time):
        """Finds the best match, prioritizing non-stale tracks."""
        best_match_id = None
        min_dist_sq = (50 * 50) # Standard max distance squared

        best_non_stale_id = None
        min_non_stale_dist_sq = min_dist_sq

        best_stale_id = None
        min_stale_dist_sq = min_dist_sq # Use same initial limit for stale, but only consider if no non-stale found

        candidates = list(self.last_positions.keys())

        for v_id in candidates:
            last_pos = self.last_positions.get(v_id)
            if not last_pos: continue

            last_seen = self.last_seen_time.get(v_id, -1)
            # --- Basic Time Check First ---
            if current_frame_time - last_seen > self.max_reid_timeout:
                  continue # Definitely too old

            miss_count = self.consecutive_misses.get(v_id, 0)
            is_stale = miss_count > self.max_misses

            dx = center_point[0]-last_pos[0]; dy = center_point[1]-last_pos[1]; dist_sq = dx*dx + dy*dy

            # --- Check against standard distance limit ---
            if dist_sq < min_dist_sq:
                if not is_stale:
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
        # (No change needed here)
        if vehicle_id not in self.vehicle_type_history: self.vehicle_type_history[vehicle_id] = {current_type_name: 1}; return current_type_name
        counts = self.vehicle_type_history[vehicle_id]; counts[current_type_name] = counts.get(current_type_name, 0) + 1
        return max(counts, key=counts.get)

    def update_track(self, vehicle_id, center_point, current_frame_time, vehicle_type_name):
        """Updates state, importantly resets consecutive misses for matched tracks."""
        self.last_positions[vehicle_id] = center_point
        self.last_seen_time[vehicle_id] = current_frame_time
        consistent_type = self.get_consistent_type(vehicle_id, vehicle_type_name)
        if vehicle_id not in self.tracking_history: self.tracking_history[vehicle_id] = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.tracking_history[vehicle_id].append(center_point)
        # --- Reset miss count on successful update/match ---
        self.consecutive_misses[vehicle_id] = 0
        return vehicle_id, consistent_type # Return ID and type used

    def increment_misses(self, ids_seen_this_frame):
        """Increments miss count for tracks not updated in the current frame."""
        missed_ids = []
        # Check all IDs currently being tracked (last_positions is a good proxy)
        for v_id in list(self.last_positions.keys()):
             if v_id not in ids_seen_this_frame:
                 self.consecutive_misses[v_id] = self.consecutive_misses.get(v_id, 0) + 1
                 missed_ids.append(v_id)
                 # Optional: Add debug print for high miss counts
                 # if self.consecutive_misses[v_id] > self.max_misses // 2:
                 #     debug_print(f"WARN: V:{v_id} miss count high: {self.consecutive_misses[v_id]}")
        return missed_ids # Return IDs that missed this frame

    def register_entry(self, vehicle_id, entry_direction, entry_time, entry_point, vehicle_type):
        if vehicle_id not in self.active_vehicles:
            self.active_vehicles[vehicle_id] = {'id': vehicle_id, 'type': vehicle_type, 'entry_direction': entry_direction,'entry_time': entry_time, 'entry_point': entry_point,'exit_direction': None, 'exit_time': None, 'exit_point': None, 'status': 'active'}
            self.consecutive_misses[vehicle_id] = 0 # Reset misses on entry
            debug_print(f"ENTRY: V:{vehicle_id} ({vehicle_type}) from {entry_direction}"); return True
        debug_print(f"WARN: V:{vehicle_id} already active, ignoring entry from {entry_direction}"); return False

    def register_exit(self, vehicle_id, exit_direction, exit_time, exit_point):
        if vehicle_id in self.active_vehicles and self.active_vehicles[vehicle_id]['status'] == 'active':
            data = self.active_vehicles[vehicle_id]
            data.update({'exit_direction': exit_direction, 'exit_time': exit_time, 'exit_point': exit_point, 'status': 'exited'})
            time_in_int = (exit_time - data['entry_time']).total_seconds(); data['time_in_intersection'] = round(time_in_int, 2)
            self.completed_paths.append(data.copy())
            debug_print(f"EXIT: V:{vehicle_id} to {exit_direction}. Time: {time_in_int:.2f}s")
            return True, time_in_int
        debug_print(f"WARN: Exit ignored for inactive/already exited V:{vehicle_id}"); return False, None

    def check_timeouts(self, current_time: datetime):
        timed_out_ids = []
        for v_id, data in list(self.active_vehicles.items()): # Use list copy
             if data['status'] == 'active':
                time_in_int = (current_time - data['entry_time']).total_seconds()
                v_type = data.get('type', 'default'); timeout = VEHICLE_TIMEOUTS.get(v_type, VEHICLE_TIMEOUTS['default'])
                if time_in_int > timeout:
                    miss_count = self.consecutive_misses.get(v_id, 0)
                    debug_print(f"TIMEOUT: V:{v_id} ({v_type}) after {time_in_int:.1f}s. Missed frames: {miss_count}")
                    data.update({'status':'timed_out', 'exit_time':current_time, 'exit_direction':'TIMEOUT', 'time_in_intersection':round(time_in_int, 2)})
                    self.completed_paths.append(data.copy()); timed_out_ids.append(v_id)
        return timed_out_ids

    def force_exit_vehicle(self, vehicle_id, exit_time: datetime):
         if vehicle_id in self.active_vehicles and self.active_vehicles[vehicle_id]['status'] == 'active':
             data = self.active_vehicles[vehicle_id]; time_in_int = (exit_time - data['entry_time']).total_seconds()
             data.update({'status':'forced_exit', 'exit_time':exit_time, 'exit_direction':'FORCED', 'time_in_intersection':round(time_in_int, 2)})
             self.completed_paths.append(data.copy()); debug_print(f"FORCED EXIT: V:{vehicle_id}"); return True
         return False

    def remove_vehicle_data(self, vehicle_id):
        """Removes all tracking data associated with a vehicle ID."""
        removed = False; pkgs = [self.active_vehicles, self.last_positions, self.last_seen_time, self.tracking_history, self.vehicle_type_history, self.consecutive_misses] # <-- Include misses
        for pkg in pkgs: removed |= (pkg.pop(vehicle_id, None) is not None)
        if removed: debug_print(f"Removed tracking data for V:{vehicle_id}")
        return removed

    def get_active_vehicle_count(self): return len(self.active_vehicles)
    def get_completed_paths(self): return self.completed_paths
    def get_tracking_trail(self, vehicle_id): return list(self.tracking_history.get(vehicle_id, []))