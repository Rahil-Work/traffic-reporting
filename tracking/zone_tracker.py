# project/tracking/zone_tracker.py
import numpy as np
import cv2
from collections import defaultdict
from config import FRAME_WIDTH, FRAME_HEIGHT # Use frame dimensions from config
from core.line_detection import Line # Assumes Line class is defined correctly
from core.utils import debug_print # For optional debug messages

class ZoneTracker:
    """Manages intersection zones and detects vehicle entries/exits."""
    def __init__(self, lines_dict):
        """
        Initializes zones based on the provided Line objects.
        Args:
            lines_dict (dict): Dictionary {'direction': Line object}
        """
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.lines = lines_dict # Keep reference
        # Call the replaced zone creation logic
        self.zones = self._create_zones_from_lines_original(lines_dict)
        self.zone_cooldowns = {}
        self.vehicle_zones = defaultdict(lambda: {'entries': set(), 'exits': set()})
        if not self.zones:
             print("WARNING: ZoneTracker initialized but no zones were created.")
        else:
             print(f"ZoneTracker initialized with {len(self.zones)} zones.")


    def _create_zones_from_lines_original(self, lines_dict):
        """
        Create entry/exit zones based on line positions using the EXACT logic
        from the user-provided original code snippet.
        """
        zones = {}
        # Use frame dimensions from config
        frame_center = (self.frame_width // 2, self.frame_height // 2)
        print("--- Creating Zones (Using ORIGINAL Logic) ---")

        # Check if lines_dict is valid
        if not lines_dict or not isinstance(lines_dict, dict):
             print("ERROR: Invalid lines_dict provided to _create_zones_from_lines_original.")
             return zones

        # First pass: Calculate line midpoints and basic perpendiculars
        line_info = {}
        print("Step 1: Calculating line properties...")
        for direction, line in lines_dict.items():
            # Ensure line is a valid Line object
            if line is None or not isinstance(line, Line):
                debug_print(f"Skipping invalid/None line for direction: {direction}")
                continue

            debug_print(f"  Processing Line [{direction.upper()}]: Points={line.points}")
            x1, y1 = line.x1, line.y1
            x2, y2 = line.x2, line.y2
            mid_x, mid_y = line.mid_x, line.mid_y
            dx, dy = line.dx, line.dy # Use pre-calculated from Line class
            length = line.length   # Use pre-calculated from Line class

            perp1 = (0, 0); perp2 = (0, 0)
            if length > 1e-6: # Avoid division by zero
                perp1 = (-dy / length, dx / length)
                perp2 = (dy / length, -dx / length)

            line_info[direction] = {
                'line': line, 'midpoint': (mid_x, mid_y),
                'perp1': perp1, 'perp2': perp2,
                'dx': dx, 'dy': dy, 'length': length
            }

        # Second pass: Determine perpendicular directions
        print("Step 2: Determining perpendicular directions (Corrected Logic)...")
        direction_pairs = [('north', 'south'), ('east', 'west')]
        processed_directions = set()

        for dir1, dir2 in direction_pairs:
            if dir1 in line_info and dir2 in line_info:
                info1, info2 = line_info[dir1], line_info[dir2]
                mid1, mid2 = info1['midpoint'], info2['midpoint']
                # Vector FROM mid1 TO mid2
                vec_x, vec_y = mid2[0] - mid1[0], mid2[1] - mid1[1]
                vec_len = np.sqrt(vec_x**2 + vec_y**2)
                if vec_len > 1e-6: vec_x /= vec_len; vec_y /= vec_len

                # Determine perpendicular for dir1 (pointing away from dir2)
                dot1_p1 = info1['perp1'][0] * vec_x + info1['perp1'][1] * vec_y
                # Choose perp with negative dot product relative to vector TO dir2
                info1['final_perp'] = info1['perp1'] if dot1_p1 < 0 else info1['perp2']

                # Determine perpendicular for dir2 (pointing away from dir1)
                # We use the vector FROM mid2 TO mid1 (-vec_x, -vec_y)
                dot2_p1 = info2['perp1'][0] * (-vec_x) + info2['perp1'][1] * (-vec_y)
                # Choose perp with negative dot product relative to vector TO dir1
                info2['final_perp'] = info2['perp1'] if dot2_p1 < 0 else info2['perp2']

                processed_directions.add(dir1); processed_directions.add(dir2)
                debug_print(f"  Relative perpendiculars set for {dir1}-{dir2} (Corrected)")


        # Corrected Fallback logic (Points AWAY from center)
        for direction, info in line_info.items():
            if direction not in processed_directions:
                debug_print(f"  Using fallback logic for {direction} (Corrected)...")
                mid_x, mid_y = info['midpoint']
                to_center_x, to_center_y = frame_center[0] - mid_x, frame_center[1] - mid_y
                to_center_len = np.sqrt(to_center_x**2 + to_center_y**2)
                if to_center_len > 1e-6: to_center_x /= to_center_len; to_center_y /= to_center_len

                # Choose perp pointing AWAY from center (negative dot product with to_center vector)
                dot1 = info['perp1'][0] * to_center_x + info['perp1'][1] * to_center_y
                # dot2 = info['perp2'][0] * to_center_x + info['perp2'][1] * to_center_y # Not needed
                info['final_perp'] = info['perp1'] if dot1 < 0 else info['perp2']
                debug_print(f"    Fallback perp chosen for {direction} (points away from center)")


        # Third pass: Create the actual zones using original point calculation
        print("Step 3: Creating zone polygons...")
        # --- Debug Image Setup ---
        debug_image = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.circle(debug_image, frame_center, 5, (0, 255, 0), -1) # Center green
        zone_colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} # BGR
        # --- End Debug Setup ---

        for direction, info in line_info.items():
            if 'final_perp' not in info:
                print(f"WARNING: Skipping zone creation for {direction}, final_perp not found.")
                continue

            line = info['line']; x1,y1=line.x1,line.y1; x2,y2=line.x2,line.y2; mid_x,mid_y=info['midpoint']
            nx, ny = info['final_perp']; dx, dy = info['dx'], info['dy']; length = info['length']
            color = zone_colors.get(direction, (255, 255, 255))

            # --- Debug Draw: Line and Perpendicular ---
            cv2.line(debug_image, (x1, y1), (x2, y2), color, 1)
            cv2.circle(debug_image, (mid_x, mid_y), 3, color, -1)
            perp_len = 30; end_x = int(mid_x + nx*perp_len); end_y = int(mid_y + ny*perp_len)
            cv2.arrowedLine(debug_image, (mid_x, mid_y), (end_x, end_y), (255, 255, 0), 2) # Yellow perp
            # --- End Debug Draw ---


            # Original calculation constants
            zone_width = 50
            zone_depth = 50
            # * Keep depth and width same to ensure zone starts from line

            # Original point calculation logic - requires length > 0 check
            if length < 1e-6:
                 print(f"WARNING: Skipping zone points for {direction} due to zero line length.")
                 continue

            try:
                p1 = (int(x1 + nx*zone_depth - dy*zone_width/length), int(y1 + ny*zone_depth + dx*zone_width/length))
                p2 = (int(x1 + nx*zone_depth + dy*zone_width/length), int(y1 + ny*zone_depth - dx*zone_width/length))
                p3 = (int(x2 + nx*zone_depth + dy*zone_width/length), int(y2 + ny*zone_depth - dx*zone_width/length))
                p4 = (int(x2 + nx*zone_depth - dy*zone_width/length), int(y2 + ny*zone_depth + dx*zone_width/length))
                zone_points = [p1, p2, p3, p4] # List of tuples
                zone_points_np = np.array(zone_points, dtype=np.int32) # Numpy array for drawing

                # --- Debug Draw: Zone Polygon ---
                cv2.polylines(debug_image, [zone_points_np], isClosed=True, color=color, thickness=1)
                # --- End Debug Draw ---

                zone_center = (mid_x + int(nx*zone_depth/2), mid_y + int(ny*zone_depth/2)) # Approx center

                zones[direction] = {
                    'points': zone_points_np, # Store numpy array
                    'perpendicular': (nx, ny),
                    'center': zone_center,
                    'line': line,
                }
                debug_print(f"  Zone created for {direction}\n Points: {zone_points_np}\n Perpendicular: {(nx, ny)}")
            except Exception as e:
                 print(f"ERROR calculating zone points for {direction}: {e}")


        # --- Save Debug Image ---
        cv2.imwrite("zone_creation_debug.png", debug_image)
        print("Saved zone_creation_debug.png")
        # --- End Save ---
        print("--- Finished Creating Zones ---")
        return zones

    # --- Other ZoneTracker methods remain the same ---
    # _check_point_in_zone, check_zone_transition, cleanup_cooldowns, remove_vehicle_data
    # (Copy these methods from the previous answer)
    def _check_point_in_zone(self, point, zone_polygon):
        if zone_polygon is None or len(zone_polygon)<3: return False
        # Ensure point is tuple of ints for pointPolygonTest
        point_int = tuple(map(int, point))
        return cv2.pointPolygonTest(zone_polygon, point_int, False) >= 0

    def check_zone_transition(self, prev_point, curr_point, vehicle_id, current_time):
        if not self.zones: return None, None
        for direction, zone_data in self.zones.items():
            poly, (nx,ny) = zone_data['points'], zone_data['perpendicular']
            prev_in, curr_in = self._check_point_in_zone(prev_point, poly), self._check_point_in_zone(curr_point, poly)
            if prev_in != curr_in: # Transition occurred
                key = f"{direction}_{vehicle_id}"; last_cross = self.zone_cooldowns.get(key, -1)
                if current_time - last_cross < 0.5: continue # Cooldown
                move_dx, move_dy = curr_point[0]-prev_point[0], curr_point[1]-prev_point[1]
                dot = move_dx*nx + move_dy*ny; event_type = None

                # Determine event based on direction dot product relative to *this zone's* perp
                # Note: The 'original' logic might make 'nx,ny' point inwards for fallback cases.
                # Entry = Moving *against* the (potentially inward) perpendicular when entering zone
                # Exit = Moving *with* the (potentially inward) perpendicular when leaving zone
                if not prev_in and curr_in:   # Moved INTO zone
                    event_type = "ENTRY" if dot < 0 else "EXIT" # Is this still right if perp points inward? Check debug image.
                elif prev_in and not curr_in: # Moved OUT OF zone
                    event_type = "EXIT" if dot > 0 else "ENTRY"

                # --- !! Potential Logic Adjustment Needed !! ---
                # If the fallback perpendicular points *inwards*, the dot product interpretation flips.
                # You might need to check if the direction used fallback logic and invert dot check.
                # This adds complexity. Simpler to use the refactored _create_zones logic if possible.
                # For now, we proceed with the current dot interpretation.

                if event_type:
                    debug_print(f"Zone: V:{vehicle_id} {event_type} Zone:{direction} Dot:{dot:.2f} Perp:({nx:.2f},{ny:.2f})")
                    self.zone_cooldowns[key] = current_time
                    if event_type == "ENTRY": self.vehicle_zones[vehicle_id]['entries'].add(direction)
                    elif event_type == "EXIT": self.vehicle_zones[vehicle_id]['exits'].add(direction)
                    return event_type, direction
        return None, None

    def cleanup_cooldowns(self, current_time):
        expired = [k for k, t in self.zone_cooldowns.items() if current_time-t >= 0.5]
        for k in expired: del self.zone_cooldowns[k]

    def remove_vehicle_data(self, vehicle_id):
        self.vehicle_zones.pop(vehicle_id, None)
        keys_to_rm = [k for k in self.zone_cooldowns if k.endswith(f"_{vehicle_id}")]
        for k in keys_to_rm: del self.zone_cooldowns[k]