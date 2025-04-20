# project/visualization/overlay.py
import cv2
import numpy as np
from core.utils import get_display_direction, debug_print
# Optional type hints if needed
# from tracking.vehicle_tracker import VehicleTracker
# from tracking.zone_tracker import ZoneTracker

# Define colors (BGR)
COLOR_GREEN = (0, 255, 0); COLOR_RED = (0, 0, 255); COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255); COLOR_MAGENTA = (255, 0, 255); COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0); COLOR_ORANGE = (0, 165, 255); COLOR_CYAN = (255, 255, 0)
ZONE_COLORS = {'north':COLOR_BLUE, 'south':COLOR_GREEN, 'east':COLOR_RED, 'west':COLOR_MAGENTA}
DEFAULT_ZONE_COLOR = COLOR_WHITE; TEXT_BG_ALPHA = 0.7

def draw_detection_box(frame, box_coords, vehicle_id, vehicle_type, status="detected", entry_dir=None, time_active=None):
    """Draws a single detection box with ID and type."""
    # Expects box object with xyxy attribute like results.boxes[i]
    # Or adapt to take simple tuple (x1,y1,x2,y2)
    if not (isinstance(box_coords, (tuple, list)) and len(box_coords) == 4):
         debug_print(f"Invalid box_coords received: {box_coords}")
         return # Cannot draw

    # Use coordinates directly
    x1, y1, x2, y2 = map(int, box_coords)

    if status == 'active': box_color = COLOR_ORANGE
    elif status == 'exiting': box_color = COLOR_RED
    elif status == 'entering': box_color = COLOR_CYAN
    else: box_color = COLOR_GREEN # Default 'detected'

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label = f"ID:{vehicle_id[-4:]} {vehicle_type}"; info_lines = [label]
    if entry_dir: info_lines.append(f"From:{get_display_direction(entry_dir)}")
    if time_active is not None: info_lines.append(f"T:{time_active:.1f}s")

    text_y = y1 - 10; max_w = 0; total_h = 0; line_h = 0
    for i, line in enumerate(info_lines):
        (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        max_w = max(max_w, w); total_h += h + (base if i==0 else 5)
        if i==0: line_h = h + base

    bg_y1 = max(0, text_y - line_h + base - 5); bg_x1 = x1
    bg_y2 = min(frame.shape[0], bg_y1 + total_h + 5); bg_x2 = min(frame.shape[1], x1 + max_w + 10)
    if bg_y1 >= bg_y2 or bg_x1 >= bg_x2: return # Avoid invalid slice

    sub_img = frame[bg_y1:bg_y2, bg_x1:bg_x2]; white_rect = np.ones(sub_img.shape, dtype=np.uint8)*255
    white_rect[:,:] = box_color; res = cv2.addWeighted(sub_img, 1.0-TEXT_BG_ALPHA, white_rect, TEXT_BG_ALPHA, 1.0)
    frame[bg_y1:bg_y2, bg_x1:bg_x2] = res

    current_y = bg_y1 + line_h - base # Start text drawing baseline
    for i, line in enumerate(info_lines):
         cv2.putText(frame, line, (x1 + 5, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 2)
         cv2.putText(frame, line, (x1 + 5, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
         if i < len(info_lines) - 1:
             (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1); current_y += h + 5

def draw_tracking_trail(frame, trail_points, vehicle_id):
    if len(trail_points) < 2: return
    try:
        color_seed = int(vehicle_id.split('_')[-1]) if '_' in vehicle_id else hash(vehicle_id)
        np.random.seed(color_seed % (2**32 - 1))
        color = tuple(np.random.randint(50, 200, size=3).tolist())
    except ValueError: color = COLOR_YELLOW
    pts = np.array(trail_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

def draw_zones(frame, zone_tracker):
    """Draws zone polygons defined in the zone_tracker."""
    # Check if zone_tracker and zones exist
    if not hasattr(zone_tracker, 'zones') or not zone_tracker.zones:
        # debug_print("draw_zones: No zones found in tracker.") # Optional debug
        return frame # Return original frame if no zones

    overlay = frame.copy(); alpha = 0.2 # For transparency

    # Iterate through the directions and the polygon point arrays
    for direction, polygon_points in zone_tracker.zones.items():

        # Ensure polygon_points is a valid NumPy array with >= 3 points
        if not isinstance(polygon_points, np.ndarray) or polygon_points.ndim != 2 or polygon_points.shape[0] < 3:
            debug_print(f"draw_zones: Skipping invalid polygon data for direction '{direction}'. Shape: {getattr(polygon_points, 'shape', 'N/A')}")
            continue

        # Ensure dtype is int32 for drawing functions
        if polygon_points.dtype != np.int32:
            polygon_points = polygon_points.astype(np.int32)

        # Get color for the zone
        color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR)

        # Draw filled polygon on overlay and outline on original frame
        try:
            cv2.fillPoly(overlay, [polygon_points], color)
            cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2)

            # Calculate centroid for placing the direction label
            M = cv2.moments(polygon_points)
            if M["m00"] != 0: # Avoid division by zero
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                # Put direction text near the calculated center
                cv2.putText(frame, direction.upper(), (center_x - 20, center_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            # else: debug_print(f"draw_zones: Could not calculate center for label '{direction}'.")

        except Exception as e:
            print(f"ERROR drawing zone for direction '{direction}': {e}")
            # Continue trying to draw other zones if one fails

    # Blend the overlay with the frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def draw_event_marker(frame, point, event_type, vehicle_id_short):
    point_int = tuple(map(int, point)); text = f"{event_type[0]}:{vehicle_id_short}"
    if event_type == "ENTRY": color = COLOR_GREEN; cv2.circle(frame, point_int, 10, color, -1)
    elif event_type == "EXIT": color = COLOR_RED; cv2.drawMarker(frame, point_int, color, cv2.MARKER_CROSS, 15, 2)
    else: return # Unknown event type
    cv2.putText(frame, text, (point_int[0]+12, point_int[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def add_status_overlay(frame, frame_number, timestamp, vehicle_tracker):
    # No change needed - uses vehicle_tracker instance
    h, w, _ = frame.shape; overlay_h = 60;
    sub_img = frame[0:overlay_h, 0:w]; black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0-TEXT_BG_ALPHA, black_rect, TEXT_BG_ALPHA, 1.0); frame[0:overlay_h, 0:w] = res
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]; active_count = vehicle_tracker.get_active_vehicle_count()
    cv2.putText(frame, f"Frame:{frame_number}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time:{time_str}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active:{active_count}", (w-180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame