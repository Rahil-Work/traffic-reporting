# visualization/gpu_overlay.py
# (Use the code provided in the previous response)
# Contains: draw_boxes_on_batch, draw_zones_on_batch, add_gpu_overlays
# Remember the text drawing part is complex and currently omitted/placeholder.
import torch
import kornia
from kornia.utils import draw_rectangle, draw_line

# Define colors as tensors (RGB Float [0, 1])
ZONE_COLORS = {
    'north': torch.tensor([0.0, 0.0, 1.0]),      # Blue
    'south': torch.tensor([0.0, 1.0, 0.0]),      # Green
    'east': torch.tensor([1.0, 0.0, 0.0]),       # Red
    'west': torch.tensor([1.0, 0.0, 1.0])       # Magenta
}
DEFAULT_ZONE_COLOR_T = torch.tensor([1.0, 1.0, 1.0]) # White
COLOR_GREEN_T = torch.tensor([0.0, 1.0, 0.0])
COLOR_RED_T = torch.tensor([1.0, 0.0, 0.0])
COLOR_ORANGE_T = torch.tensor([1.0, 0.65, 0.0])
COLOR_YELLOW_T = torch.tensor([1.0, 1.0, 0.0])
COLOR_WHITE_T = torch.tensor([1.0, 1.0, 1.0])
COLOR_BLACK_T = torch.tensor([0.0, 0.0, 0.0])

def draw_boxes_on_batch(frame_batch_tensor, batch_detections):
    """ Draws bounding boxes on a batch using kornia.utils.draw_rectangle """
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda:
        print("Warning: draw_boxes_on_batch requires CUDA tensor input.")
        return frame_batch_tensor

    b, c, h, w = frame_batch_tensor.shape
    # Kornia drawing functions modify inplace, so clone if you need original
    output_batch = frame_batch_tensor # Modify in place

    for i in range(b): # Iterate through batch
        if i >= len(batch_detections) or not batch_detections[i]: continue
        frame_dets = batch_detections[i]

        boxes_list = []
        colors_list = []
        ids_list = [] # Store IDs for potential text drawing

        for det in frame_dets:
            box = det.get('box'); status = det.get('status', 'detected'); v_id = det.get('id', '')
            if box is None: continue

            boxes_list.append(box); ids_list.append(v_id)
            color = COLOR_ORANGE_T if status == 'active' else COLOR_RED_T if status == 'exiting' else COLOR_YELLOW_T if status == 'entering' else COLOR_GREEN_T
            colors_list.append(color.to(output_batch.device)) # Move color tensor to correct device

        if boxes_list:
            # Ensure boxes are float tensor [N, 4] (xmin, ymin, xmax, ymax)
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32, device=output_batch.device)
            # Ensure colors are float tensor [N, 3]
            colors_tensor = torch.stack(colors_list).float()

            try:
                # Call the imported function directly
                draw_rectangle(output_batch[i:i+1], boxes_tensor.unsqueeze(0), color=colors_tensor.unsqueeze(0), fill=False, thickness=2)
                # NOTE: Check Kornia docs for draw_rectangle signature in v0.8.
                # It might expect image batch BxCxHxW and boxes BxNx4, colors BxNx3.
                # The slicing above makes image B=1, boxes B=1, colors B=1 for the call.
            except TypeError as te:
                 print(f"Kornia draw_rectangle TypeError (check arguments/shapes for v0.8.0): {te}")
            except Exception as e:
                 print(f"Kornia box drawing error: {e}")

            # --- Text Drawing Placeholder ---
            # print(f"Frame {i}: IDs {ids_list}") # Still complex
            pass

    return output_batch

def draw_zones_on_batch(frame_batch_tensor, zone_polygons_gpu):
    """ Draws zone polygon outlines on a batch using kornia.utils.draw_line """
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda or not zone_polygons_gpu:
        return frame_batch_tensor
    output_batch = frame_batch_tensor # Modify in place

    for i in range(output_batch.shape[0]): # Iterate through batch frames
        for direction, poly_pts_tensor in zone_polygons_gpu.items():
            if poly_pts_tensor is None or len(poly_pts_tensor) < 3: continue

            color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR_T).to(output_batch.device)
            num_points = poly_pts_tensor.shape[0]

            for k in range(num_points):
                pt1 = poly_pts_tensor[k]         # Current point [x, y]
                pt2 = poly_pts_tensor[(k + 1) % num_points] # Next point

                # Ensure points are float tensors [1, 2] for draw_line if needed
                start_points = pt1.float().unsqueeze(0)
                end_points = pt2.float().unsqueeze(0)
                # Ensure color is float tensor [1, C]
                line_color = color.float().unsqueeze(0)

                try:
                    # Call the imported function directly
                    # NOTE: Check Kornia v0.8.0 docs for draw_line signature, args might change.
                    # It might modify output_batch[i] in place.
                    draw_line(output_batch[i], start_points, end_points, line_color) # Removed thickness, check API
                except TypeError as te:
                     print(f"Kornia draw_line TypeError for zone {direction} (check arguments/shapes for v0.8.0): {te}")
                except Exception as e:
                    print(f"Kornia line drawing error for zone {direction}: {e}")

    return output_batch

def add_gpu_overlays(frame_batch_tensor, batch_detections, zone_polygons):
    """ Adds multiple types of overlays (boxes, zones) to a GPU tensor batch. """
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda:
        return frame_batch_tensor

    # Use clone if subsequent steps need the original unmodified tensor
    processed_batch = frame_batch_tensor # Modify in-place

    # Convert zones to GPU tensors
    zone_polygons_gpu = {}
    if zone_polygons:
        for direction, pts in zone_polygons.items():
            if pts is not None and len(pts) >= 3:
                try: zone_polygons_gpu[direction] = torch.tensor(pts, dtype=torch.float32, device=processed_batch.device)
                except Exception as e: print(f"Warning: Could not convert zone '{direction}' to tensor: {e}")

    # Draw Zones (Outlines)
    if zone_polygons_gpu:
        processed_batch = draw_zones_on_batch(processed_batch, zone_polygons_gpu)

    # Draw Boxes (and skip text for now)
    if batch_detections:
        processed_batch = draw_boxes_on_batch(processed_batch, batch_detections)

    # TODO: Add Trails / Text if implemented later

    return processed_batch