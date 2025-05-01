# project/tracking/zone_tracker.py
import numpy as np
import cv2
from collections import defaultdict
from config import FRAME_WIDTH, FRAME_HEIGHT # Use frame dimensions from config
from core.line_detection import Line # Assumes Line class is defined correctly
from core.utils import debug_print # For optional debug messages

class ZoneTracker:
    """Manages intersection zones and detects vehicle entries/exits."""
    def __init__(self, polygons_dict):
        """
        Initializes zones based on the provided Line objects.
        Args:
            lines_dict (dict): Dictionary {'direction': Line object}
        """
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.zones = polygons_dict if polygons_dict else {}
        self.zone_cooldowns = {}
        self.vehicle_zones = {}
        print(f"ZoneTracker __init__: Received polygons_dict: {polygons_dict}")
        if not self.zones:
            print("ZoneTracker __init__: Received empty or invalid polygons_dict.")
        else:
            print(f"ZoneTracker initialized with {len(self.zones)} zones: {list(self.zones.keys())}")

    def _check_point_in_zone(self, point, zone_polygon):
        if zone_polygon is None or len(zone_polygon) < 3:
            # print(f"DEBUG ZoneCheck: Invalid polygon received: {zone_polygon}") # Optional debug
            return False
        try:
            # Ensure point is tuple of standard Python ints for pointPolygonTest
            point_int = tuple(map(int, point))
            # Ensure polygon is numpy array of int32
            if not isinstance(zone_polygon, np.ndarray) or zone_polygon.dtype != np.int32:
                zone_polygon = np.array(zone_polygon, dtype=np.int32) # Attempt conversion if needed

            result = cv2.pointPolygonTest(zone_polygon, point_int, False)
            # result >= 0 means inside or on the boundary
            return result >= 0
        except Exception as e:
            print(f"ERROR in _check_point_in_zone: {e}, Point: {point}, Polygon shape: {getattr(zone_polygon, 'shape', 'N/A')}, Polygon dtype: {getattr(zone_polygon, 'dtype', 'N/A')}")
            return False

    def check_zone_transition(self, prev_center_point, current_bbox, vehicle_id, current_time):
        """
        Checks if a vehicle transitioned into or out of any defined zone polygon.
        Uses previous center point and current bottom-center point.
        Args:
            prev_center_point (tuple): The (x, y) coordinates of the vehicle's center in the previous frame.
            current_bbox (tuple): The (x1, y1, x2, y2) coordinates of the vehicle's current bounding box.
            vehicle_id (str): The ID of the vehicle.
            current_time (float): The current frame time in seconds.
        """
        if not self.zones: return None, None
        # Ensure we have the necessary points/box
        if prev_center_point is None or current_bbox is None: return None, None

        # Calculate bottom-center point from the current bounding box
        x1, y1, x2, y2 = current_bbox
        current_bottom_center = ((x1 + x2) // 2, y2) # Use bottom-y (y2)

        # Debug Print (Optional)
        # print(f"DEBUG ZoneCheck V:{vehicle_id}: PrevCenter={prev_center_point}, CurrBC={current_bottom_center}")

        for direction, poly_array in self.zones.items():
            if poly_array is None or len(poly_array) < 3: continue

            # Check previous center point against the polygon
            prev_in = self._check_point_in_zone(prev_center_point, poly_array)
            # Check current bottom-center point against the polygon
            curr_in = self._check_point_in_zone(current_bottom_center, poly_array)

            # Debug Print (Optional)
            # print(f"DEBUG ZoneCheck V:{vehicle_id} Dir:{direction}: PrevIn(Center)={prev_in}, CurrIn(BottomCenter)={curr_in}")

            if prev_in != curr_in: # A transition occurred
                key = f"{direction}_{vehicle_id}"; last_cross = self.zone_cooldowns.get(key, -1)
                on_cooldown = (current_time - last_cross < 0.1)

                # Debug Print (Optional)
                # print(f"DEBUG_EXIT_CHECK: V:{vehicle_id} Dir:{direction} Transition! PrevIn={prev_in} CurrIn={curr_in} CooldownActive={on_cooldown}")

                if on_cooldown: continue # Still on cooldown

                event_type = None
                if not prev_in and curr_in: event_type = "ENTRY"
                elif prev_in and not curr_in: event_type = "EXIT"

                if event_type:
                    # print(f"***** Zone Transition Detected (BottomCenter) ***** V:{vehicle_id} Dir:{direction} Type:{event_type} *****")
                    # print(f"  PrevCenter: {prev_center_point} (In: {prev_in}), CurrBC: {current_bottom_center} (In: {curr_in})") # Optional

                    self.zone_cooldowns[key] = current_time
                    if vehicle_id not in self.vehicle_zones: self.vehicle_zones[vehicle_id] = {'entries': set(), 'exits': set()}
                    if event_type == "ENTRY": self.vehicle_zones[vehicle_id]['entries'].add(direction)
                    elif event_type == "EXIT": self.vehicle_zones[vehicle_id]['exits'].add(direction)

                    return event_type, direction

        return None, None

    def cleanup_cooldowns(self, current_time):
        expired = [k for k, t in self.zone_cooldowns.items() if current_time-t >= 0.5]
        for k in expired: del self.zone_cooldowns[k]

    def remove_vehicle_data(self, vehicle_id):
        if vehicle_id in self.vehicle_zones:
            self.vehicle_zones.pop(vehicle_id, None)
        keys_to_rm = [k for k in self.zone_cooldowns if k.endswith(f"_{vehicle_id}")]
        for k in keys_to_rm: del self.zone_cooldowns[k]