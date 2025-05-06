# visualization/gpu_overlay.py
# (Use the code provided in the previous response)
# Contains: draw_boxes_on_batch, draw_zones_on_batch, add_gpu_overlays
# Remember the text drawing part is complex and currently omitted/placeholder.
import torch
import kornia.draw as KDraw

# Define colors as tensors (RGB Float [0, 1])
ZONE_COLORS = { # Add more colors as needed
    'north': torch.tensor([0.0, 0.0, 1.0]), 'south': torch.tensor([0.0, 1.0, 0.0]),
    'east': torch.tensor([1.0, 0.0, 0.0]), 'west': torch.tensor([1.0, 0.0, 1.0]) }
DEFAULT_ZONE_COLOR_T = torch.tensor([1.0, 1.0, 1.0]) # White
COLOR_GREEN_T = torch.tensor([0.0, 1.0, 0.0]); COLOR_RED_T = torch.tensor([1.0, 0.0, 0.0])
COLOR_ORANGE_T = torch.tensor([1.0, 0.65, 0.0]); COLOR_YELLOW_T = torch.tensor([1.0, 1.0, 0.0])

def draw_boxes_on_batch(frame_batch_tensor, batch_detections):
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda: return frame_batch_tensor
    b, c, h, w = frame_batch_tensor.shape
    output_batch = frame_batch_tensor # Draw in-place ok? Kornia might. Let's try.

    for i in range(b):
        if i >= len(batch_detections) or not batch_detections[i]: continue
        frame_dets = batch_detections[i]
        boxes_list, colors_list, ids_list = [], [], []

        for det in frame_dets:
            box, status, v_id = det.get('box'), det.get('status', 'detected'), det.get('id', '')
            if box is None: continue
            boxes_list.append(box); ids_list.append(v_id)
            color = COLOR_ORANGE_T if status == 'active' else COLOR_RED_T if status == 'exiting' else COLOR_YELLOW_T if status == 'entering' else COLOR_GREEN_T
            colors_list.append(color.to(output_batch.device))

        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32, device=output_batch.device)
            colors_tensor = torch.stack(colors_list)
            try: # Kornia modifies tensor in-place
                KDraw.draw_rectangle(output_batch[i], boxes_tensor, color=colors_tensor, fill=False, thickness=2)
            except Exception as e: print(f"Kornia box drawing error: {e}")
            # --- Placeholder for Text/ID drawing ---
            # print(f"Frame {i} IDs: {ids_list}") # Debug
    return output_batch

def draw_zones_on_batch(frame_batch_tensor, zone_polygons_gpu):
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda or not zone_polygons_gpu:
        return frame_batch_tensor
    output_batch = frame_batch_tensor # Modify in-place

    for i in range(output_batch.shape[0]):
        for direction, poly_pts_tensor in zone_polygons_gpu.items():
            if poly_pts_tensor is None or len(poly_pts_tensor) < 3: continue
            color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR_T).to(output_batch.device)
            num_points = poly_pts_tensor.shape[0]
            for k in range(num_points):
                pt1 = poly_pts_tensor[k]; pt2 = poly_pts_tensor[(k + 1) % num_points]
                try: # Kornia modifies tensor in-place
                    KDraw.draw_line(output_batch[i], pt1.unsqueeze(0), pt2.unsqueeze(0), color.unsqueeze(0), thickness=2)
                except Exception as e: print(f"Kornia line drawing error for zone {direction}: {e}")
    return output_batch

def add_gpu_overlays(frame_batch_tensor, batch_detections, zone_polygons):
    if not isinstance(frame_batch_tensor, torch.Tensor) or not frame_batch_tensor.is_cuda: return frame_batch_tensor

    processed_batch = frame_batch_tensor # Work in-place

    # Convert zones to GPU tensors once per call
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

    # TODO: Add Trails if needed (complex)
    # TODO: Add Status Text if needed (very complex)

    return processed_batch