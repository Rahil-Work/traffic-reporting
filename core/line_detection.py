# project/core/line_detection.py
import numpy as np
import cv2

class Line:
    """Represents a detection line with basic properties."""
    def __init__(self, points, direction):
        if not points or len(points) != 2:
            raise ValueError("Line requires exactly two points.")
        try:
            self.x1, self.y1 = map(int, points[0])
            self.x2, self.y2 = map(int, points[1])
        except (TypeError, ValueError) as e:
             raise ValueError(f"Invalid point format in points list {points}: {e}")

        self.points = [(self.x1, self.y1), (self.x2, self.y2)]
        self.direction = direction.lower()

        self.mid_x = (self.x1 + self.x2) // 2
        self.mid_y = (self.y1 + self.y2) // 2
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.length = np.sqrt(self.dx**2 + self.dy**2)

        # Cooldown tracking (remains useful even if zones are primary)
        self.crossing_cooldown = {}
        self.cooldown_time = 0.5 # Seconds

    # Note: check_crossing_simple might not be used if ZoneTracker is primary,
    # but keep it for potential future use or debugging.
    def check_crossing_simple(self, point1, point2):
        """Basic geometric intersection check."""
        x1, y1 = point1
        x2, y2 = point2
        lx1, ly1 = self.x1, self.y1
        lx2, ly2 = self.x2, self.y2

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            return 0 if val == 0 else (1 if val > 0 else 2)

        o1 = orientation((x1, y1), (x2, y2), (lx1, ly1))
        o2 = orientation((x1, y1), (x2, y2), (lx2, ly2))
        o3 = orientation((lx1, ly1), (lx2, ly2), (x1, y1))
        o4 = orientation((lx1, ly1), (lx2, ly2), (x2, y2))

        return o1 != o2 and o3 != o4 # General case intersection

    def apply_cooldown(self, vehicle_id, current_time):
        self.crossing_cooldown[vehicle_id] = current_time

    def is_on_cooldown(self, vehicle_id, current_time):
        last_time = self.crossing_cooldown.get(vehicle_id, -1)
        return (current_time - last_time) < self.cooldown_time

    def cleanup_cooldowns(self, current_time):
        """Removes expired cooldown entries."""
        expired_ids = [
            v_id for v_id, last_time in self.crossing_cooldown.items()
            if current_time - last_time >= self.cooldown_time
        ]
        for v_id in expired_ids:
            del self.crossing_cooldown[v_id]
        return len(expired_ids)

# Helper function to draw lines (could be moved to visualization)
def draw_lines_on_image(image, lines_dict):
    """Draws Line objects from a dictionary onto an image."""
    if image is None or not lines_dict: return image
    img_copy = image.copy()
    colors = {
        'north': (255, 0, 0), 'south': (0, 255, 0),
        'east': (0, 0, 255), 'west': (255, 0, 255)
    }
    default_color = (255, 255, 255)

    for direction, line_obj in lines_dict.items():
        if line_obj and isinstance(line_obj, Line): # Ensure it's a Line object
            color = colors.get(direction, default_color)
            cv2.line(img_copy, line_obj.points[0], line_obj.points[1], color, 2)
            cv2.putText(img_copy, direction.upper(),
                        (line_obj.mid_x, line_obj.mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_copy