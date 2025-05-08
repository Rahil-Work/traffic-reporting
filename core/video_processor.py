# project/core/video_processor.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import threading
import queue
import shutil # For directory cleanup
from datetime import datetime, timedelta
import concurrent.futures
import re
import kornia.color as K

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, DEBUG_MODE, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION,
    # New Chunking/Encoding Config
    ENABLE_AUTO_CHUNKING, AUTO_CHUNK_THRESHOLD_MINUTES,
    AUTO_CHUNK_DURATION_MINUTES, CHUNK_TEMP_DIR, REPORT_TEMP_DIR,
    FFMPEG_PATH, ENCODER_CODEC, ENCODER_BITRATE, ENCODER_PRESET,
    FINAL_VIDEO_EXTENSION, REPORT_OUTPUT_DIR,
    TRACKER_TYPE
)

# Modules
from core.utils import (debug_print, format_timestamp, is_valid_movement,
                    cleanup_memory, get_video_properties, split_video_ffmpeg)
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device

# Import tracker based on config OR default to VehicleTracker
if TRACKER_TYPE == 'Custom':
    from tracking.vehicle_tracker import VehicleTracker

from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data

# Conditional GPU Accelerated Components
GPU_VIZ_ENABLED = False
NvidiaFrameReader = None
NvidiaFrameWriter = None
add_gpu_overlays = None

try:
    from core.nvidia_reader import NvidiaFrameReader
    print("Imported NvidiaFrameReader.")
except ImportError as e_reader:
    print(f"Warning: Failed to import NvidiaFrameReader ({e_reader}). GPU accelerated reading disabled.")
    NvidiaFrameReader = None # Ensure it's None if import fails

# --- Conditional import for Writer and Overlay based on Visualization ---
if ENABLE_VISUALIZATION:
    if NvidiaFrameReader is not None: # Only try writer/overlay if reader worked
        try:
            from visualization.gpu_overlay import add_gpu_overlays
            from core.nvidia_writer import NvidiaFrameWriter
            GPU_VIZ_ENABLED = True
            print("GPU Writer & Overlay components imported successfully.")
        except ImportError as e_viz:
            import traceback; traceback.print_exc()
            print(f"Warning: Failed to import GPU Writer/Overlay components ({e_viz}).")
            print("Visualization will be disabled.")
            GPU_VIZ_ENABLED = False
            add_gpu_overlays = None
            NvidiaFrameWriter = None
            # Import CPU fallback overlay if needed elsewhere
            # from visualization import overlay as visual_overlay
    else:
        # Reader failed to import, so GPU viz is impossible
        print("NvidiaFrameReader failed to import, disabling GPU Visualization.")
        GPU_VIZ_ENABLED = False
        # Import CPU fallback overlay if needed elsewhere
        # from visualization import overlay as visual_overlay
else:
    print("ENABLE_VISUALIZATION is False. No video output will be generated.")
    # Import CPU fallback overlay if needed elsewhere
    # from visualization import overlay as visual_overlay

USE_CPU_READER_FALLBACK = False # Set to True if you want cv2.VideoCapture as fallback
if NvidiaFrameReader is None and not USE_CPU_READER_FALLBACK:
     # Raise an error or exit if Nvidia reading is essential
     raise ImportError("NvidiaFrameReader failed to import and CPU fallback is disabled.")
elif NvidiaFrameReader is None and USE_CPU_READER_FALLBACK:
     print("Using CPU Frame Reader (cv2.VideoCapture) as fallback.")
     # Ensure the rest of the code handles the case where self.reader_instance is None
     # and uses self._frame_reader_task (CPU version) instead.
     # This requires changes in _process_single_video_file to switch reader logic.

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance

from reporting.excel_report import create_excel_report, consolidate_excel_reports

class VideoProcessor:
    def __init__(self):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE
        self.max_workers = THREAD_COUNT

        # Gradio Polygon State
        self.gradio_polygons = {} # Stores {direction: List[tuple(x,y)]}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.model_names = {} # Store model class names

        self.tracker_type_internal = TRACKER_TYPE
        self.tracker_lock = threading.Lock() # Might still be useful for custom tracker logic

        # Attributes managed per file/chunk run by _process_single_video_file
        self.tracker = None
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.frame_read_queue = None
        self.video_write_queue = None # For CPU VideoWriter (old) or NvidiaFrameWriter's input
        self.stop_event = None
        self.reader_thread = None
        self.writer_instance = None # Can be NvidiaFrameWriter or old CPU writer thread
        self.writer_thread = None # For CPU writer
        self.last_cleanup_time = 0
        self.final_output_path_current_file = None # Path for video output of the current file/chunk

        # CUDA Streams
        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0

        # Load model once
        try:
            print("VideoProcessor: Initializing Model...")
            self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError("Model loading failed during init.")
            # Store model names if available after loading
            self.model_names = getattr(self.model, 'names', {})
            if not self.model_names:
                print("VideoProcessor: Warning - Could not get class names from loaded model.")
            print("VideoProcessor: Model loaded successfully in __init__.")
        except Exception as e:
            print(f"FATAL: Error loading model during VideoProcessor init: {e}")
            self.model = None # Indicate model loading failure

    # --- Gradio Polygon Related Methods ---

    def _draw_polygons_for_gradio(self, frame_to_draw_on):
        if frame_to_draw_on is None: return None
        img_copy = frame_to_draw_on.copy()
        colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} # BGR
        default_color = (255, 255, 255); temp_color = (0, 255, 255)

        # Draw completed polygons
        for direction, points in self.gradio_polygons.items():
            if points and len(points) >= 3:
                pts_np = np.array(points, dtype=np.int32)
                color = colors.get(direction, default_color)
                cv2.polylines(img_copy, [pts_np], isClosed=True, color=color, thickness=2)
                try:
                    if len(pts_np) > 0:
                        M = cv2.moments(pts_np)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                            cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass # Ignore label drawing errors

        # Draw temporary points
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2: # Closing hint
                last_pt = self.gradio_temp_points[-1]; first_pt = self.gradio_temp_points[0]
                hint_color = tuple(c // 2 for c in current_temp_color)
                cv2.line(img_copy, last_pt, first_pt, hint_color, 1, cv2.LINE_AA)

        self.last_drawn_polygons_frame = img_copy
        return img_copy

    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive': return None, "Zone definition mode is 'hardcoded'. Drawing disabled."
        self.reset_gradio_polygons()
        if video_path is None: return None, "Please upload a video."
        try:
            cap=cv2.VideoCapture(video_path); ret,frame=cap.read(); cap.release()
            if ret:
                frame = cv2.resize(frame, self.frame_shape)
                self.original_frame_for_gradio = frame.copy()
                drawn_frame = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
                return drawn_frame, "Video loaded. Select direction and click image to define zone vertices."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"

    def set_gradio_polygon_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        direction_lower = direction.lower()
        if direction_lower in self.gradio_polygons:
             del self.gradio_polygons[direction_lower]
             print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower
        self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1])
        self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3:
            status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."
            frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
            return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy()
        status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else:
            removed_point = self.gradio_temp_points.pop()
            status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def reset_gradio_polygons(self):
        self.gradio_polygons = {}; self.current_gradio_direction = None
        self.gradio_temp_points = []; status = "All polygons cleared."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio) if self.original_frame_for_gradio is not None else None
        self.last_drawn_polygons_frame = frame_to_show
        return frame_to_show, status

    # --- Video Reading/Writing Threads ---
    def _frame_reader_task(self, video_path, original_fps):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print("ERROR: Reader failed to open video."); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps)); frame_count = 0; frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read();
                if not ret: break
                if frame_count % frame_interval == 0:
                    frame_time_seconds = frame_count / original_fps
                    try:
                        # Add frame to queue for processing
                        self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                        frames_yielded += 1
                    except queue.Full:
                        debug_print("Reader queue full, waiting...")
                        time.sleep(0.1);
                        if self.stop_event.is_set(): break
                        try: # Retry put
                            self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                            frames_yielded += 1
                        except queue.Full: print("ERROR: Reader queue persistently full."); break
                    except Exception as e: debug_print(f"Reader queue error: {e}"); break
                frame_count += 1
            debug_print(f"Reader finished. Read {frame_count}, yielded ~{frames_yielded}.")
        except Exception as e: print(f"ERROR in reader thread: {e}")
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, timeout=1.0) # Signal end to reader queue
                except queue.Full: pass

    def _frame_writer_task(self, output_path):
        # This task will only run if ENABLE_VISUALIZATION is True
        out = None; writer_initialized = False; frames_written = 0
        actual_output_path = output_path
        try:
            # Attempt to initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI.");
                avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID');
                out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened():
                    actual_output_path = avi_path;
                    # Important: Update the class variable if the path changed
                    self.final_output_path = avi_path
                else:
                    print(f"ERROR: Writer failed to open for both MP4 and AVI.");
                    # Signal failure (e.g., by setting final_output_path to None or an error string)
                    self.final_output_path = "ERROR_WRITER_FAILED"
                    return # Exit thread if writer cannot be opened

            writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")

            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0) # Wait for frames
                except queue.Empty:
                    # Check if processing is done and queue is empty
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue # Continue waiting if not stopped

                if frame_data is None: break # Check for sentinel value

                frame = frame_data
                # Perform checks and write frame
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    # Ensure frame matches output shape before writing
                    if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]:
                        frame = cv2.resize(frame, self.frame_shape)
                    out.write(frame); frames_written += 1

                # Mark task as done only if queue exists
                if self.video_write_queue: self.video_write_queue.task_done()

        except Exception as e:
            print(f"ERROR in writer thread: {e}")
            # Signal failure
            if self.final_output_path != "ERROR_WRITER_FAILED": # Avoid overwriting specific error
                 self.final_output_path = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            # Release writer if it was initialized
            if out and writer_initialized:
                 out.release();
                 debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            # Signal writer queue end only if it exists (ensures main thread doesn't block indefinitely)
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass


    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, current_batch_of_gpu_tensors_from_reader):
        """
        Prepares a batch for model inference and visualization.
        Input: List of raw GPU tensors from NvidiaFrameReader (expected to represent NV12).
        Output: ARGB batch tensor [B, 4, H_model, W_model], float, range [0,1]
                (where H_model, W_model are MODEL_INPUT_SIZE)
        """
        if not current_batch_of_gpu_tensors_from_reader:
            log_prefix = getattr(self, 'thread_name_prefix', 'Preproc')
            print(f"{log_prefix}: Empty input tensor list.")
            return None
        
        processed_argb_tensors_for_batch = []
        target_h_model, target_w_model = self.model_input_size_config, self.model_input_size_config
        log_prefix = getattr(self, 'thread_name_prefix', 'Preproc') # For logging consistency

        # Ensure reader dimensions are available (should be set by now)
        if not (hasattr(self.reader_instance, 'width') and self.reader_instance.width > 0 and \
                hasattr(self.reader_instance, 'height') and self.reader_instance.height > 0):
            print(f"{log_prefix}: Reader dimensions not yet available for NV12 conversion. Cannot process batch.")
            return None
        
        original_h = self.reader_instance.height
        original_w = self.reader_instance.width

        # Check if dimensions are divisible by 2 (required by yuv420_to_rgb)
        if original_h % 2 != 0 or original_w % 2 != 0:
            print(f"{log_prefix}: CRITICAL - Original video dimensions ({original_w}x{original_h}) "
                  f"are not evenly divisible by 2. Kornia YUV420 conversion will likely fail. "
                  f"Consider preprocessing videos to have even dimensions or adding padding/cropping.")
            # Depending on requirements, you might try to crop here, but it's complex. Best to fix input videos.
            # Returning None for now.
            return None

        for idx, raw_nv12_tensor in enumerate(current_batch_of_gpu_tensors_from_reader):
            if raw_nv12_tensor is None: continue
            
            if str(raw_nv12_tensor.device) != str(self.device):
                raw_nv12_tensor = raw_nv12_tensor.to(self.device)

            # --- NV12 to RGB Conversion using kornia.color.yuv420_to_rgb ---
            # Debug print for the first frame's input tensor
            if idx == 0: 
                print(f"Debug Preproc (Frame 0): raw_nv12_tensor shape: {raw_nv12_tensor.shape}, dtype: {raw_nv12_tensor.dtype}")

            # Step 1: Validate shape and dtype (expecting uint8, (H*1.5, W) from dlpack)
            if raw_nv12_tensor.ndim != 2 or raw_nv12_tensor.shape != (int(original_h * 1.5), original_w):
                print(f"{log_prefix}: Unexpected raw_nv12_tensor shape: {raw_nv12_tensor.shape} for frame index {idx}. "
                      f"Expected (H*1.5, W) = ({int(original_h * 1.5)}, {original_w}). Skipping frame.")
                continue
                
            if raw_nv12_tensor.dtype != torch.uint8:
                print(f"{log_prefix}: Warning (frame index {idx}) - NV12 tensor dtype is {raw_nv12_tensor.dtype}, expected torch.uint8. Skipping frame (conversion may be needed earlier or check reader).")
                # It's crucial that the input tensor is uint8 [0,255] before separating planes.
                continue

            # Step 2: Separate Y and UV planes
            try:
                # Y plane is the first H rows
                y_plane = raw_nv12_tensor[:original_h, :] # Shape: [H, W]
                # UV plane is the remaining H/2 rows, containing interleaved UV data
                uv_plane_interleaved = raw_nv12_tensor[original_h:, :] # Shape: [H/2, W]

                # Step 3: De-interleave UV plane and reshape
                # NV12 format: U V U V ...
                # Reshape to [H/2, W/2, 2] -> channel dimension stores U and V
                uv_plane_deinterleaved = uv_plane_interleaved.reshape(original_h // 2, original_w // 2, 2)
                
                # Separate U and V. Kornia expects UV together as [B, 2, H/2, W/2]
                # Permute to [2, H/2, W/2]
                uv_plane_permuted = uv_plane_deinterleaved.permute(2, 0, 1) # Shape: [2, H/2, W/2]

                # Step 4: Add Batch and Channel dimensions, convert Y to float [0,1], UV to float [-0.5, 0.5]
                # Kornia's yuv420_to_rgb expects Y in [0,1] and UV in [-0.5, 0.5]
                y_plane_final = y_plane.float().unsqueeze(0).unsqueeze(0) / 255.0 # Shape: [1, 1, H, W], Range [0,1]
                
                # Convert UV uint8 [0,255] to float [-0.5, 0.5] range expected by Kornia
                # Formula: float = (uint8 / 255.0) - 0.5 (approximately, exact range might depend on standard)
                # Or more precisely: Y range 16-235, UV range 16-240 -> map to float ranges.
                # Let's use the simple scaling first, Kornia might handle standard ranges internally.
                # uv_plane_final = (uv_plane_permuted.float() / 255.0) - 0.5 # Incorrect range mapping potentially
                
                # Kornia often works with YCbCr values directly. Let's try passing uint8 Y and uint8 UV
                # and see if yuv420_to_rgb handles the conversion internally, OR convert to float and scale later.
                # Trying with float inputs scaled simply:
                # Y: [0, 1]
                # UV: Convert uint8 [0, 255] to float [0, 1] first, Kornia might handle the range shift.
                # Or check docs if it expects specific integer ranges.
                # Safest bet might be to convert RGB->YUV420 in Kornia to see expected input ranges.
                
                # Let's assume yuv420_to_rgb takes Y [0,1] and UV [0,1] (from uint8/255) and handles offsets internally.
                y_plane_input = y_plane.unsqueeze(0).unsqueeze(0).float() / 255.0 # [1, 1, H, W]
                uv_plane_input = uv_plane_permuted.unsqueeze(0).float() / 255.0 # [1, 2, H/2, W/2]

            except Exception as e_reshape:
                print(f"{log_prefix}: Error separating/reshaping YUV planes for frame index {idx}: {e_reshape}. Skipping frame.")
                continue

            # Step 5: Convert YUV420 to RGB using Kornia
            try:
                frame_tensor_rgb_k = K.yuv420_to_rgb(y_plane_input, uv_plane_input) # Input: Y[B,1,H,W], UV[B,2,H/2,W/2] (float [0,1])
                # Output: [B, 3, H, W], float32 [0,1]
                if frame_tensor_rgb_k is None: raise RuntimeError("kornia.color.yuv420_to_rgb returned None")
                frame_tensor_rgb = frame_tensor_rgb_k.squeeze(0) # Remove batch dim -> [3, H_original, W_original]

            except Exception as e_conv:
                print(f"{log_prefix}: Error during yuv420_to_rgb conversion for frame index {idx}: {e_conv}. Skipping frame.")
                import traceback; traceback.print_exc() # Print stack trace for Kornia errors
                continue

            # frame_tensor_rgb is now [3, H_original, W_original], float32, range [0,1]

            # Step 6: Add Alpha channel to make it ARGB (Alpha first)
            _c, current_h, current_w = frame_tensor_rgb.shape
            alpha_channel = torch.ones((1, current_h, current_w), dtype=frame_tensor_rgb.dtype, device=frame_tensor_rgb.device)
            # Create ARGB: Alpha, R, G, B
            frame_tensor_argb = torch.cat((alpha_channel, frame_tensor_rgb), dim=0) # [4, H_original, W_original]

            # Step 7: Resize to model input size
            if current_h != target_h_model or current_w != target_w_model:
                try:
                    frame_tensor_argb_resized = F.interpolate(
                        frame_tensor_argb.unsqueeze(0), size=(target_h_model, target_w_model),
                        mode='bilinear', align_corners=False
                    ).squeeze(0) # [4, H_model, W_model]
                except Exception as e_resize:
                    print(f"{log_prefix}: Error during resize for frame index {idx}: {e_resize}. Skipping frame.")
                    continue
            else:
                frame_tensor_argb_resized = frame_tensor_argb
            
            processed_argb_tensors_for_batch.append(frame_tensor_argb_resized)

        # --- Batch Post-processing ---
        if not processed_argb_tensors_for_batch:
            print(f"{log_prefix}: No tensors were successfully processed in this batch.")
            return None
            
        # Stack processed tensors into a batch
        # Output is ARGB, float32, [0,1], resized to model_input_size
        final_batch_tensor_argb = torch.stack(processed_argb_tensors_for_batch)
        
        return final_batch_tensor_argb

    def _process_inference_results(self, results_yolo, input_batch_tensor_for_viz, frame_numbers, frame_times, start_datetime_chunk):
        """
        Processes YOLO results for a batch, runs tracking, zone checks.
        Returns:
            - batch_structured_log_data (list of dicts): For Excel reporting.
            - batch_detections_for_gpu_viz (list of lists): Data for gpu_overlay.add_gpu_overlays.
                                                        Each inner list corresponds to a frame and contains dicts:
                                                        {'box': [x1,y1,x2,y2], 'id': 'v_xx', 'type': 'Car', 'status': 'active'}
        """
        batch_size_actual = len(results_yolo)
        batch_structured_log_data = [[] for _ in range(batch_size_actual)]
        batch_detections_for_gpu_viz = [[] for _ in range(batch_size_actual)]

        def process_single_frame_logic(idx):
            result = results_yolo[idx]
            frame_number_abs = frame_numbers[idx] # Absolute frame number in original video
            frame_time_sec_abs = frame_times[idx] # Time offset from original video start
            current_timestamp_abs = start_datetime_chunk + timedelta(seconds=frame_time_sec_abs) # Absolute timestamp

            frame_log_entries = []  # For this frame's Excel log data
            frame_viz_entries = []  # For this frame's GPU visualization data

            if not self.tracker or not self.zone_tracker:
                return [], []

            # 1. Extract Detections from YOLO results
            detections_for_tracker = [] # List of {'box_coords': (x1o,y1o,x2o,y2o), 'center':pt, 'type':name}
            # Assuming results are scaled to the input_batch_tensor size (model_input_size_config)
            tensor_h, tensor_w = input_batch_tensor_for_viz.shape[2], input_batch_tensor_for_viz.shape[3]

            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    v_type_name = self.model_names.get(cls_id, f"CLS_{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # Ensure coords are within tensor bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(tensor_w - 1, x2), min(tensor_h - 1, y2)
                    
                    center_pt = ((x1 + x2) // 2, (y1 + y2) // 2) # Center for matching
                    bottom_center_pt = ((x1 + x2) // 2, y2) # Bottom-center for zone check convention

                    detections_for_tracker.append({
                        'box_coords': (x1, y1, x2, y2),
                        'center': center_pt,
                        'bottom_center': bottom_center_pt, # For zone check logic
                        'type': v_type_name,
                        'confidence': float(box.conf[0])
                    })

            # --- Existing Custom Tracker Logic (adapted) ---
            ids_seen_this_frame_custom = set()
            processed_tracks_this_frame_map = {} # {v_id: {data for viz and zone check}}

            if self.tracker_type_internal == 'Custom': # Assuming self.tracker is VehicleTracker instance
                # Match existing tracks
                matched_ids_map = {} # det_idx -> v_id
                unmatched_detections_indices = []
                for det_idx, det_info in enumerate(detections_for_tracker):
                    # Use 'center' or 'bottom_center' for find_best_match based on its implementation
                    # Your VehicleTracker.find_best_match expects center_point and frame_time_sec
                    matched_id = self.tracker.find_best_match(det_info['center'], frame_time_sec_abs)
                    if matched_id: matched_ids_map[det_idx] = matched_id
                    else: unmatched_detections_indices.append(det_idx)

                # Update matched tracks
                for det_idx, v_id in matched_ids_map.items():
                    det_info = detections_for_tracker[det_idx]
                    v_id, consistent_type = self.tracker.update_track(
                        v_id, det_info['center'], frame_time_sec_abs, det_info['type']
                    )
                    processed_tracks_this_frame_map[v_id] = {
                        'box': det_info['box_coords'], 'type': consistent_type, 'id': v_id,
                        'center_for_zone': det_info['bottom_center'], # Use bottom-center for zone check
                        'event': None, 'status': 'detected'
                    }
                    ids_seen_this_frame_custom.add(v_id)

                # Handle new tracks
                for det_idx in unmatched_detections_indices:
                    det_info = detections_for_tracker[det_idx]
                    v_id = self.tracker._get_next_id() # Get new ID from tracker
                    v_id, consistent_type = self.tracker.update_track(
                        v_id, det_info['center'], frame_time_sec_abs, det_info['type']
                    )
                    processed_tracks_this_frame_map[v_id] = {
                        'box': det_info['box_coords'], 'type': consistent_type, 'id': v_id,
                        'center_for_zone': det_info['bottom_center'],
                        'event': None, 'status': 'new'
                    }
                    ids_seen_this_frame_custom.add(v_id)
            
            # --- Zone Transition Logic (Using data from processed_tracks_this_frame_map) ---
            ids_to_remove_this_frame = set()
            active_zone_keys = list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []

            for v_id, track_data in processed_tracks_this_frame_map.items():
                current_bbox_for_zone = track_data['box'] # This is the (x1,y1,x2,y2) from detection
                # For prev_pos, VehicleTracker.tracking_history stores center points usually
                # ZoneTracker.check_zone_transition expects prev_center_point and current_bbox
                prev_pos = None
                if v_id in self.tracker.tracking_history and len(self.tracker.tracking_history[v_id]) > 1:
                    prev_pos = self.tracker.tracking_history[v_id][-2] # This is a center point

                if prev_pos and self.zone_tracker and self.zone_tracker.zones:
                    if self.perf: self.perf.start_timer('zone_checking')
                    event_type, event_dir = self.zone_tracker.check_zone_transition(
                        prev_pos, current_bbox_for_zone, v_id, frame_time_sec_abs # Use current frame time (absolute from video start)
                    )
                    if self.perf: self.perf.end_timer('zone_checking')

                    v_state = self.tracker.active_vehicles.get(v_id) # State from VehicleTracker
                    consistent_type = track_data['type'] # Type determined by tracker

                    if event_type == "ENTRY":
                        # Your existing ENTRY and Inferred EXIT logic here, adapted slightly:
                        # - Use current_timestamp_abs for event times
                        # - Use consistent_type from track_data
                        # - Append to frame_log_entries
                        # - Update processed_tracks_this_frame_map[v_id]['event'] / ['status']
                        if v_state and v_state.get('status') == 'active': # Potentially an inferred exit
                            stored_entry_dir = v_state.get('entry_direction')
                            if stored_entry_dir and stored_entry_dir != event_dir and is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                final_type_inf = self.tracker.get_consistent_type(v_id, consistent_type) # Get final type before exit
                                success_inf, t_in_int_inf = self.tracker.register_exit(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'])
                                if success_inf:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_inf)
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': final_type_inf, 'event_type': 'EXIT', 'direction_from': stored_entry_dir, 'direction_to': event_dir, 'status': 'exited_inferred', 'time_in_intersection': t_in_int_inf, 'frame_number': frame_number_abs})
                                    ids_to_remove_this_frame.add(v_id)
                                    processed_tracks_this_frame_map[v_id]['event'] = 'EXIT' # Mark for drawing
                                # Now, register the new entry
                                if self.tracker.register_entry(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'], consistent_type):
                                    if self.perf: self.perf.record_vehicle_entry()
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': consistent_type, 'event_type': 'ENTRY', 'direction_from': event_dir, 'direction_to': None, 'status': 'entry', 'frame_number': frame_number_abs})
                                    processed_tracks_this_frame_map[v_id]['event'] = 'ENTRY' # Override for drawing
                                    processed_tracks_this_frame_map[v_id]['status'] = 'active'
                            # else: Re-entry or invalid inferred, handled by standard entry if needed
                        
                        # Standard new entry (or if inferred exit didn't apply and it's still not active)
                        if not (v_state and v_state.get('status') == 'active'): # If not already active from this entry dir
                            if self.tracker.register_entry(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'], consistent_type):
                                if self.perf: self.perf.record_vehicle_entry()
                                frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': consistent_type, 'event_type': 'ENTRY', 'direction_from': event_dir, 'direction_to': None, 'status': 'entry', 'frame_number': frame_number_abs})
                                processed_tracks_this_frame_map[v_id]['event'] = 'ENTRY'
                                processed_tracks_this_frame_map[v_id]['status'] = 'active'


                    elif event_type == "EXIT":
                        # Your existing EXIT logic here:
                        # - Use current_timestamp_abs, consistent_type
                        # - Append to frame_log_entries
                        # - Add to ids_to_remove_this_frame
                        # - Update processed_tracks_this_frame_map[v_id]['event'] / ['status']
                        if v_state and v_state.get('status') == 'active':
                            entry_dir_trk = v_state.get('entry_direction')
                            if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir, active_zone_keys):
                                final_exit_type = self.tracker.get_consistent_type(v_id, consistent_type)
                                success_ex, t_in_int_ex = self.tracker.register_exit(v_id, event_dir, current_timestamp_abs, track_data['center_for_zone'])
                                if success_ex:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_ex)
                                    frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id, 'vehicle_type': final_exit_type, 'event_type': 'EXIT', 'direction_from': entry_dir_trk, 'direction_to': event_dir, 'status': 'exit', 'time_in_intersection': t_in_int_ex, 'frame_number': frame_number_abs})
                                    ids_to_remove_this_frame.add(v_id)
                                    processed_tracks_this_frame_map[v_id]['event'] = 'EXIT'
                                    processed_tracks_this_frame_map[v_id]['status'] = 'exited'

            # --- Timeout Check (using self.tracker) ---
            timed_out_ids = self.tracker.check_timeouts(current_timestamp_abs)
            for v_id_to in timed_out_ids:
                ids_to_remove_this_frame.add(v_id_to)
                if self.perf: self.perf.record_vehicle_exit('timed_out')
                p_data = next((p for p in reversed(self.tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'), None)
                if p_data:
                     frame_log_entries.append({'timestamp_dt': current_timestamp_abs, 'vehicle_id': v_id_to, 'vehicle_type': p_data.get('type', 'UnknownType'), 'event_type': 'TIMEOUT', 'direction_from': p_data.get('entry_direction', 'UNKNOWN'), 'direction_to': 'TIMEOUT', 'status': 'timeout', 'time_in_intersection': p_data.get('time_in_intersection', 'N/A'), 'frame_number': frame_number_abs})
                if v_id_to in processed_tracks_this_frame_map:
                    processed_tracks_this_frame_map[v_id_to]['status'] = 'timed_out'


            # --- Update Misses & Remove data (using self.tracker) ---
            if self.tracker_type_internal == 'Custom':
                self.tracker.increment_misses(ids_seen_this_frame_custom)
            for v_id_rem in ids_to_remove_this_frame:
                self.tracker.remove_vehicle_data(v_id_rem)
                if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_rem)


            # --- Prepare data for GPU visualization ---
            current_active_vehicles_state = self.tracker.active_vehicles.copy()
            for v_id, data in processed_tracks_this_frame_map.items():
                if v_id not in ids_to_remove_this_frame: # Only visualize if still relevant
                    viz_status = 'detected' # Default
                    vehicle_state_for_viz = current_active_vehicles_state.get(v_id)
                    if vehicle_state_for_viz and vehicle_state_for_viz.get('status') == 'active':
                        viz_status = 'active'
                    elif data.get('event') == 'EXIT': # If explicitly exited this frame
                        viz_status = 'exiting'
                    elif data.get('event') == 'ENTRY' and not (vehicle_state_for_viz and vehicle_state_for_viz.get('status') == 'active'):
                         viz_status = 'entering' # New entry this frame

                    frame_viz_entries.append({
                        'box': data['box'], # Already scaled to model_input_size
                        'id': data['id'],
                        'type': data['type'],
                        'status': viz_status
                        # Add more if gpu_overlay.py needs it (e.g., trail points if you implement GPU trails)
                    })
            return frame_log_entries, frame_viz_entries
            # --- End process_single_frame_logic ---

        # --- Parallel Execution for the batch ---
        if self.perf: self.perf.start_timer('detection_processing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_frame_logic, i): i for i in range(batch_size_actual)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    frame_log, frame_viz_data = future.result()
                    batch_structured_log_data[idx] = frame_log
                    batch_detections_for_gpu_viz[idx] = frame_viz_data
                except Exception as exc:
                    print(f'\nError in post-processing frame {frame_numbers[idx]} (batch idx {idx}): {exc}')
                    import traceback; traceback.print_exc()
                    batch_structured_log_data[idx] = [] # Ensure lists are populated even on error
                    batch_detections_for_gpu_viz[idx] = []
        if self.perf: self.perf.end_timer('detection_processing')

        flat_log_data = [item for sublist in batch_structured_log_data for item in sublist]
        return flat_log_data, batch_detections_for_gpu_viz # batch_detections_for_gpu_viz is list of lists

    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        """
        Orchestrates video processing, handling chunking and consolidation.
        This function is now a GENERATOR, yielding status updates.
        The final yield will be a dict: {'final_message': "..."}
        """
        if not self.processing_lock.acquire(blocking=False):
            yield {'final_message': "Processing is already in progress."}
            return

        yield "Orchestrator: Initializing processing..."
        print(f"--- Orchestrator: Starting Video Processing ---"); print(f"Video: {video_path}")
        
        final_outcome_message = "Processing did not complete as expected." # Default

        # Ensure model is loaded (if not done in __init__ or if it needs re-checking)
        if self.model is None:
            yield "Orchestrator: Loading model..."
            print("Orchestrator: Model is None, attempting to load...")
            try:
                self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
                if self.model is None: raise RuntimeError("Model loading returned None in orchestrator.")
                self.model_names = getattr(self.model, 'names', {})
                yield "Orchestrator: Model loaded successfully."
            except Exception as e:
                error_msg = f"❌ FATAL: Error loading model: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}
                return

        def parse_dt(date_str, time_str): # Helper
            try:
                dt_str = f"{date_str}{time_str[:6]}"
                base_dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")
                return base_dt.replace(microsecond=int(time_str[6:].ljust(6, '0')))
            except Exception as e:
                print(f"Error parsing datetime: {date_str} {time_str} - {e}")
                return None
        
        original_start_dt = parse_dt(start_date_str, start_time_str)
        if not original_start_dt:
            error_msg = "❌ Error: Invalid start date/time format."
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}
            return

        yield "Orchestrator: Reading video properties..."
        duration_sec, _, _ = get_video_properties(video_path) # Assuming get_video_properties is robust
        if duration_sec is None:
            error_msg = f"❌ Error: Cannot read video properties: {video_path}"
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}
            return

        should_chunk = ENABLE_AUTO_CHUNKING and duration_sec > (AUTO_CHUNK_THRESHOLD_MINUTES * 60)
        chunk_files = [video_path] # Default to original file
        chunk_duration_sec_cfg = AUTO_CHUNK_DURATION_MINUTES * 60

        if should_chunk:
            status_msg_chunking = f"Orchestrator: Video duration ({duration_sec:.0f}s) triggers chunking (~{AUTO_CHUNK_DURATION_MINUTES} min chunks)."
            yield status_msg_chunking
            print(status_msg_chunking)
            
            # Define a simple callback for ffmpeg splitting status
            def ffmpeg_split_progress(progress_val, message_val): # progress_val not used yet by split_video_ffmpeg
                yield message_val # Yield the message from ffmpeg_split

            try:
                # This progress_callback for split_video_ffmpeg needs to be handled by that function
                # to yield messages if we want live ffmpeg progress (complex).
                # For now, split_video_ffmpeg has print statements and its own progress_callback parameter.
                # The one passed here from Gradio won't be directly used by ffmpeg's stdout.
                # Let's simplify: split_video_ffmpeg will print, and we yield before/after.
                yield "Orchestrator: Splitting video now... (this may take a while)"
                chunk_files = split_video_ffmpeg(video_path, CHUNK_TEMP_DIR, chunk_duration_sec_cfg, FFMPEG_PATH, progress_callback=None) # Pass None for now
                if not chunk_files: raise RuntimeError("FFmpeg splitting yielded no files.")
                yield f"Orchestrator: Video split into {len(chunk_files)} chunks."
            except Exception as e:
                error_msg = f"❌ Error during video splitting: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}
                return
            
            if os.path.exists(REPORT_TEMP_DIR): shutil.rmtree(REPORT_TEMP_DIR) # Clean old temp reports
            os.makedirs(REPORT_TEMP_DIR, exist_ok=True)
        else:
            status_msg_single = f"Orchestrator: Processing as single file (Duration: {duration_sec:.0f}s)."
            yield status_msg_single
            print(status_msg_single)

        all_chunk_processing_ok = True
        num_chunks = len(chunk_files)
        processed_chunk_results = [] # To store result_msg_part from each chunk

        for i, current_chunk_path in enumerate(chunk_files):
            chunk_start_offset = timedelta(seconds=i * chunk_duration_sec_cfg if should_chunk else 0)
            current_chunk_start_dt = original_start_dt + chunk_start_offset
            current_start_date_str_chunk = current_chunk_start_dt.strftime("%y%m%d")
            current_start_time_str_chunk = current_chunk_start_dt.strftime("%H%M%S") + f"{current_chunk_start_dt.microsecond // 1000:03d}"

            status_processing_chunk = f"Orchestrator: Starting segment {i+1}/{num_chunks}: {os.path.basename(current_chunk_path)}"
            yield status_processing_chunk
            print("*" * 60); print(status_processing_chunk)

            temp_report_override_path = None
            if should_chunk:
                base_chunk_name = os.path.splitext(os.path.basename(current_chunk_path))[0]
                temp_report_override_path = os.path.join(REPORT_TEMP_DIR, f"report_{base_chunk_name}.xlsx")

            # --- Define a callback for _process_single_video_file to yield its internal progress ---
            def single_file_progress_callback(p_value, status_message_internal):
                # This callback is called from within _process_single_video_file's loop
                # We need to yield this status message out to Gradio
                yield f"Chunk {i+1}/{num_chunks} - {status_message_internal}" # Prepend chunk info
            
            # _process_single_video_file is NOT a generator, so we can't loop over its yields here.
            # The progress_callback passed to it would need to use a queue or shared state if we
            # want live updates from its internal loop in Gradio without making it a generator too.
            # For now, the status updates from _process_single_video_file will primarily be to the console.
            # The yields here are for *orchestrator-level* status.
            
            result_msg_part = self._process_single_video_file(
                video_path=current_chunk_path,
                start_date_str=current_start_date_str_chunk,
                start_time_str=current_start_time_str_chunk,
                primary_direction=primary_direction,
                output_path_override=temp_report_override_path,
                progress_callback=None, # Pass None; _process_single_video_file prints console progress
                chunk_info=(i + 1, num_chunks)
            )
            processed_chunk_results.append(result_msg_part) # Store individual result

            if "❌ Error" in result_msg_part or "FATAL" in result_msg_part:
                all_chunk_processing_ok = False
                final_outcome_message = result_msg_part # Store the first critical error
                error_stop_msg = f"Orchestrator: Error processing {os.path.basename(current_chunk_path)}. Halting."
                yield error_stop_msg
                print(error_stop_msg)
                break # Stop processing further chunks
            else:
                yield f"Orchestrator: Finished segment {i+1}/{num_chunks}."


        print("*" * 60)
        # --- Final Outcome / Consolidation ---
        if all_chunk_processing_ok:
            if should_chunk:
                consolidation_start_msg = "Orchestrator: Consolidating reports..."
                yield consolidation_start_msg
                print(consolidation_start_msg)

                original_base = os.path.splitext(os.path.basename(video_path))[0]
                final_consolidated_report_path = os.path.join(REPORT_OUTPUT_DIR, f"detection_logs_{original_base}_CONSOLIDATED.xlsx")
                os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
                
                consolidated_path = consolidate_excel_reports(REPORT_TEMP_DIR, final_consolidated_report_path, original_start_dt.date())
                
                if consolidated_path:
                    # Combine individual chunk summaries for a more complete message
                    full_summary = "Chunk Processing Summaries:\n" + "\n\n".join(processed_chunk_results)
                    full_summary += f"\n\n✅ Processing completed.\nConsolidated report: {consolidated_path}"
                    final_outcome_message = full_summary
                    yield "Orchestrator: Report consolidation successful."
                    # Optional: Clean up temp report dir
                    shutil.rmtree(REPORT_TEMP_DIR)
                else:
                    final_outcome_message = f"⚠️ Processing finished, but report consolidation failed. Chunk reports in {REPORT_TEMP_DIR}."
                    yield "Orchestrator: Report consolidation failed."
            else: # Single file success
                final_report_path_single = None
                # Use the result from the single processed file
                result_msg_part_single = processed_chunk_results[0] if processed_chunk_results else ""
                
                match = re.search(r"Report segment: (.+\.xlsx)", result_msg_part_single) or \
                        re.search(r"Excel report: (.+\.xlsx)", result_msg_part_single)
                if match: final_report_path_single = match.group(1)

                if final_report_path_single and os.path.exists(final_report_path_single):
                    # For single file, the result_msg_part_single already contains the full summary.
                    final_outcome_message = result_msg_part_single
                    # No need to add "Report: ..." as it's in result_msg_part_single
                else:
                    final_outcome_message = f"⚠️ Processing finished for single file, but report file not found. Message: {result_msg_part_single}"

        elif not final_outcome_message: # Error occurred, but message wasn't set by the loop
            final_outcome_message = "❌ Processing stopped due to errors in chunk processing."

        if should_chunk:
            cleanup_msg = "Orchestrator: Cleaning up temporary video chunks..."
            yield cleanup_msg
            print(cleanup_msg)
            try:
                shutil.rmtree(CHUNK_TEMP_DIR)
                yield "Orchestrator: Temporary video chunks cleaned up."
            except Exception as e:
                warn_cleanup_msg = f"Warning: Could not remove temp chunk dir: {e}"
                yield warn_cleanup_msg
                print(warn_cleanup_msg)

        release_msg = "Orchestrator: All processing stages finished. Releasing lock."
        yield release_msg
        print(release_msg)
        if self.processing_lock.locked(): self.processing_lock.release()

        # The VERY LAST yield is the final package for Gradio's handle_process
        yield {'final_message': final_outcome_message}

    # --- Core Processing Logic for ONE file/chunk ---
    def _process_single_video_file(self, video_path, start_date_str, start_time_str, primary_direction, output_path_override=None, progress_callback=None, chunk_info=(1,1)):
        """ Processes a single video file (original or chunk) using Nvidia Reader/Writer if enabled. """
        base_video_name = os.path.basename(video_path)
        print(f"_process_single_video_file: Initializing for {base_video_name}")

        # Determine if GPU acceleration is enabled for this run
        _viz_enabled_this_run = ENABLE_VISUALIZATION and GPU_VIZ_ENABLED
        writer_initialized_this_run = False # Track if writer init was attempted

        # --- Per-File/Chunk State Initialization ---
        self.tracker = None # Initialize tracker instance
        self.zone_tracker = None # Initialize zone tracker instance
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS) # New performance tracker
        self.stop_event = threading.Event() # Event to signal threads to stop
        self.reader_dimensions_ready_event = threading.Event() # Event for reader->main sync
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None # Zones used for this run
        self.writer_instance = None # Holds NvidiaFrameWriter instance
        self.reader_instance = None # Holds NvidiaFrameReader instance
        self.reader_thread = None # Holds reader thread object
        self.final_output_path_current_file = None # Path to the final muxed video for this file/chunk
        self.frame_read_queue = None # Queue for reader output

        try:
            # --- Parse Start Datetime for this chunk/file ---
            try:
                dt_str = f"{start_date_str}{start_time_str[:6]}"
                base_dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")
                start_datetime_chunk = base_dt.replace(microsecond=int(start_time_str[6:].ljust(6, '0')))
                print(f"_process_single_video_file: Parsed start datetime: {start_datetime_chunk.isoformat()}")
            except ValueError as e_dt:
                 raise ValueError(f"Invalid start date/time format '{start_date_str} {start_time_str}': {e_dt}") from e_dt

            # --- Zone Setup (Hardcoded or Interactive) ---
            print(f"_process_single_video_file: Setting up zones (mode: {self.line_mode})...")
            if self.line_mode == 'hardcoded':
                 if not LINE_POINTS: raise ValueError("Config LINE_POINTS missing for hardcoded mode.")
                 valid_polys = {k: v for k, v in LINE_POINTS.items() if isinstance(v, list) and len(v) >= 3}
                 if not valid_polys: raise ValueError("No valid polygons found in config LINE_POINTS.")
                 self.detection_zones_polygons = {k: np.array(v, dtype=np.int32) for k, v in valid_polys.items()}
            elif self.line_mode == 'interactive':
                 if not self.gradio_polygons: raise ValueError("Gradio polygons not defined for interactive mode.")
                 valid_polys = {k: v for k, v in self.gradio_polygons.items() if v and len(v) >= 3}
                 if len(valid_polys) < 2: raise ValueError("Need at least 2 valid Gradio zones defined.")
                 self.detection_zones_polygons = {k: np.array(v, dtype=np.int32) for k, v in valid_polys.items()}
            else:
                 raise ValueError(f"Invalid LINE_MODE configured: {self.line_mode}")

            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("ZoneTracker initialization failed.")
            print(f"ZoneTracker initialized with {len(self.zone_tracker.zones)} zones: {list(self.zone_tracker.zones.keys())}")

            # --- Initialize Tracker ---
            print(f"_process_single_video_file: Initializing tracker (type: {self.tracker_type_internal})...")
            if self.tracker_type_internal == 'Custom':
                self.tracker = VehicleTracker() # Use configured parameters if needed
            # Add elif blocks for other tracker types here
            else:
                raise NotImplementedError(f"Tracker type {self.tracker_type_internal} not fully implemented in _process_single_video_file.")
            if self.tracker is None: raise RuntimeError("Tracker initialization failed.")
            print("Tracker initialized.")


            # --- Get Video Properties for the current file ---
            print(f"_process_single_video_file: Getting video properties for {base_video_name}...")
            duration_sec_file, original_fps_file, total_frames_file = get_video_properties(video_path)
            if duration_sec_file is None: raise FileNotFoundError(f"Cannot open or read video properties for: {video_path}")
            if original_fps_file <= 0:
                print(f"Warning: Original FPS read as {original_fps_file}. Using default 30.0.")
                original_fps_file = 30.0
            print(f"Video Properties: Duration={duration_sec_file:.2f}s, OrigFPS={original_fps_file:.2f}, TotalFrames~={total_frames_file}")

            # --- Setup and Start Reader Thread (NvidiaFrameReader) ---
            print(f"_process_single_video_file: Setting up NvidiaFrameReader...")
            self.frame_read_queue = queue.Queue(maxsize=self.batch_size * 4) # Queue for (tensor, frame_num, timestamp)
            self.reader_instance = NvidiaFrameReader(
                video_path=video_path,
                target_fps=self.target_fps,
                original_fps_hint=original_fps_file,
                frame_queue=self.frame_read_queue,
                stop_event=self.stop_event,
                dimensions_ready_event=self.reader_dimensions_ready_event, # Pass the event
                device_id_for_torch_output=0 # Assuming device 0
                # Pass cuda_context_handle/cuda_stream_handle if managed externally
            )
            self.reader_thread = threading.Thread(target=self.reader_instance.run, name=f"NvidiaReader-{base_video_name}", daemon=True)
            self.reader_thread.start()
            print("NvidiaReader thread started. Main loop will wait for dimensions event before initializing writer.")

            # --- Main Processing Loop ---
            all_detections_structured_log = [] # Stores dicts for Excel report for this file/chunk
            frames_processed_count = 0          # Frames processed by inference/tracking
            if self.perf: self.perf.start_processing(total_frames_file)

            # Batch accumulation lists
            frames_batch_gpu_tensors = [] # Holds raw tensors from reader queue
            frame_numbers_batch_abs = []  # Absolute frame numbers corresponding to tensors
            frame_times_batch_abs = []    # Absolute time offsets corresponding to tensors

            last_progress_print_time = time.monotonic()
            current_chunk_idx_disp, total_chunks_disp = chunk_info # For display

            processed_frame_indices_log = []

            print(f"_process_single_video_file: Entering main processing loop for {base_video_name}...")
            while True: # Loop until reader queue signals end (None)
                try:
                    # Get data from the reader thread's queue
                    frame_data = self.frame_read_queue.get(timeout=120) # Long timeout
                    
                    if frame_data is None: # Check for EOS sentinel from reader
                        if self.frame_read_queue: self.frame_read_queue.task_done() # Mark None as processed
                        print(f"_process_single_video_file: Received EOS (None) from reader queue. Exiting processing loop.")
                        break # Exit main loop

                    # Unpack the data from the queue
                    gpu_tensor_from_reader, frame_num_abs, frame_time_abs = frame_data
                    if gpu_tensor_from_reader is None: # Should not happen if reader sends valid data or None
                        if self.frame_read_queue: self.frame_read_queue.task_done()
                        print(f"Warning: Received None tensor from reader queue unexpectedly (Frame {frame_num_abs}). Skipping.")
                        continue

                    # --- LAZY WRITER INITIALIZATION (using Event) ---
                    # Attempt to initialize the writer only once if visualization is enabled
                    if _viz_enabled_this_run and self.writer_instance is None and not writer_initialized_this_run:
                        print(f"_process_single_video_file: Checking if writer needs initialization (Frame {frame_num_abs})...")
                        # Wait for the reader to signal that dimensions are ready (or that it failed)
                        print(f"Waiting for reader dimensions_ready_event...")
                        event_was_set = self.reader_dimensions_ready_event.wait(timeout=10.0) # Wait up to 10s

                        if event_was_set:
                            # Event was set, check if reader successfully got dimensions
                            reader_w = getattr(self.reader_instance, 'width', 0)
                            reader_h = getattr(self.reader_instance, 'height', 0)

                            if reader_w > 0 and reader_h > 0:
                                print(f"Video dimensions ready: {reader_w}x{reader_h}. Initializing NvidiaFrameWriter...")
                                chunk_video_base = f"output_{os.path.splitext(base_video_name)[0]}"
                                final_muxed_path = os.path.join(VIDEO_OUTPUT_DIR, f"{chunk_video_base}{FINAL_VIDEO_EXTENSION}")
                                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
                                
                                # Determine format writer should expect (AFTER overlay drawing)
                                # _preprocess_batch outputs ARGB float, overlay draws on it, then convert to UINT8 ARGB for writer
                                writer_input_format = "ARGB" # String format name for PyNvVideoCodec

                                self.writer_instance = NvidiaFrameWriter(
                                    output_path=final_muxed_path, width=reader_w, height=reader_h,
                                    fps=self.target_fps,
                                    encoder_codec=ENCODER_CODEC, bitrate=ENCODER_BITRATE, preset=ENCODER_PRESET,
                                    input_tensor_format_str=writer_input_format,
                                    temp_dir=CHUNK_TEMP_DIR, device_id=0 # Assuming device 0 for encoder
                                )
                                self.writer_instance.start() # Starts writer thread, which calls _initialize_encoder

                                # Check if the internal encoder object was created successfully
                                if hasattr(self.writer_instance, 'encoder') and self.writer_instance.encoder is not None:
                                    writer_initialized_this_run = True # Mark as successfully initialized
                                    print("Nvidia Writer thread started successfully.")
                                    self.final_output_path_current_file = final_muxed_path # Store potential output path
                                else:
                                    print("NvidiaWriter CRITICAL: _initialize_encoder failed inside writer thread. Writing disabled.")
                                    self.writer_instance = None # Set back to None to prevent usage
                                    writer_initialized_this_run = True # Mark as attempted (and failed)
                            else:
                                print(f"Reader dimensions event was set, but reader width/height still 0 (w={reader_w},h={reader_h}). "
                                      "Reader initialization or parameter update likely failed. Writing disabled.")
                                writer_initialized_this_run = True # Mark as attempted (and failed)
                        else:
                            print("Timeout waiting for reader dimensions_ready_event. Reader might be stuck or very slow. "
                                  "Writing disabled for this run.")
                            writer_initialized_this_run = True # Mark as attempted (and failed due to timeout)
                    
                    # --- Accumulate Batch ---
                    frames_batch_gpu_tensors.append(gpu_tensor_from_reader)
                    frame_numbers_batch_abs.append(frame_num_abs)
                    frame_times_batch_abs.append(frame_time_abs)

                    # --- Process Batch when Full ---
                    if len(frames_batch_gpu_tensors) >= self.batch_size:
                        processed_frame_indices_log.extend(frame_numbers_batch_abs)
                        batch_start_mono = time.monotonic()
                        # 1. Preprocess (NV12 -> ARGB float [0,1], resize)
                        if self.perf: self.perf.start_timer('preprocessing')
                        batch_argb_for_viz_and_model = self._preprocess_batch(frames_batch_gpu_tensors)
                        if self.perf: self.perf.end_timer('preprocessing')

                        batch_log_entries = []
                        batch_viz_data_for_gpu = []
                        num_processed_in_batch = len(frames_batch_gpu_tensors) # Store length before clear

                        if batch_argb_for_viz_and_model is not None and batch_argb_for_viz_and_model.nelement() > 0:
                            # 2. Extract RGB for Model
                            yolo_model_input = batch_argb_for_viz_and_model[:, 1:4, :, :] # Select R, G, B

                            # <<< Debug: Print tensor stats for a specific frame >>>
                            TARGET_DEBUG_FRAME = 100 # Example frame number
                            pipeline_name = "GPU" # Change to "CPU" when adding to CPU pipeline code
                            if TARGET_DEBUG_FRAME in frame_numbers_batch_abs:
                                try:
                                    batch_index = frame_numbers_batch_abs.index(TARGET_DEBUG_FRAME)
                                    tensor_to_debug = yolo_model_input[batch_index].detach().cpu() # Move to CPU for printing stats
                                    print(f"\n--- Input Tensor Stats for Frame {TARGET_DEBUG_FRAME} ({pipeline_name} Pipeline) ---")
                                    print(f"  Shape: {tensor_to_debug.shape}")
                                    print(f"  Dtype: {tensor_to_debug.dtype}")
                                    # Calculate stats carefully to avoid large tensor ops if possible
                                    t_min = tensor_to_debug.min().item()
                                    t_max = tensor_to_debug.max().item()
                                    t_mean = tensor_to_debug.mean().item()
                                    t_std = tensor_to_debug.std().item()
                                    print(f"  Min:   {t_min:.6f}")
                                    print(f"  Max:   {t_max:.6f}")
                                    print(f"  Mean:  {t_mean:.6f}")
                                    print(f"  Std:   {t_std:.6f}")
                                    # Print a small slice (e.g., Top-left 3x3 pixels, all channels)
                                    print("  Slice (Top-left 3x3, Channels 0,1,2):")
                                    print(tensor_to_debug[:, :3, :3])
                                    print("--- End Tensor Stats ---")
                                except Exception as e_stat:
                                    print(f"\nError getting tensor stats: {e_stat}")
                            
                            input()


                            # 3. Inference
                            stream = None
                            if self.device == 'cuda' and self.streams: stream = self.streams[self.current_stream_index % len(self.streams)]; self.current_stream_index += 1
                            with torch.cuda.stream(stream) if stream else torch.no_grad():
                                with torch.amp.autocast(device_type=self.device, enabled=MIXED_PRECISION and self.device == 'cuda'):
                                    if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream)
                                    print(f"Debug VideoProcessor: Shape passed to model: {yolo_model_input.shape}, Dtype: {yolo_model_input.dtype}")
                                    yolo_results = self.model(yolo_model_input, verbose=False) # Pass RGB
                                    if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                            if stream: stream.synchronize()
                            if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                            # 4. Post-processing (Tracking, Zone Checks)
                            # Pass RGB tensor results were based on, frame numbers/times, start datetime
                            batch_log_entries, batch_viz_data_for_gpu = self._process_inference_results(
                                yolo_results, yolo_model_input, frame_numbers_batch_abs, frame_times_batch_abs, start_datetime_chunk
                            )
                            all_detections_structured_log.extend(batch_log_entries)
                            if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in yolo_results if hasattr(r,'boxes') and r.boxes))

                            # 5. GPU Visualization & Writing
                            if _viz_enabled_this_run and self.writer_instance is not None: # Check writer is valid
                                if self.perf: self.perf.start_timer('drawing_gpu')
                                # Draw on the ARGB tensor
                                visualized_batch_argb_float = add_gpu_overlays(
                                    batch_argb_for_viz_and_model, batch_viz_data_for_gpu, self.zone_tracker.zones
                                )
                                if self.perf: self.perf.end_timer('drawing_gpu')

                                if visualized_batch_argb_float is not None:
                                    for frame_tensor_argb_float in visualized_batch_argb_float: # This is [4,H,W] ARGB float [0,1]
                                        # Convert float [0,1] ARGB to uint8 [0,255] ARGB for NvidiaFrameWriter "ARBG" input
                                        frame_to_encode_argb_uint8 = (frame_tensor_argb_float.clamp(0,1) * 255.0).byte()
                                        # Ensure channel order didn't change if writer strictly needs A first
                                        self.writer_instance.put(frame_to_encode_argb_uint8)

                        else: # Preprocessing failed for the batch
                            print(f"Warning: Preprocessing returned None or empty tensor for batch starting frame {frame_numbers_batch_abs[0]}. Skipping inference for this batch.")


                        # --- Batch Post-Processing Steps ---
                        batch_time_taken = time.monotonic() - batch_start_mono
                        frames_processed_count += num_processed_in_batch
                        if self.perf: self.perf.record_batch_processed(num_processed_in_batch, batch_time_taken); self.perf.sample_system_metrics()

                        # --- Periodic Cleanup ---
                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                             if self.perf: self.perf.start_timer('memory_cleanup')
                             last_frame_time_in_batch = frame_times_batch_abs[-1] if frame_times_batch_abs else 0
                             cleanup_tracking_data(self.tracker, self.zone_tracker, last_frame_time_in_batch)
                             cleanup_memory() # General torch/cuda cleanup
                             self.last_cleanup_time = current_mono_time
                             if self.perf: self.perf.end_timer('memory_cleanup')

                        # --- Progress Reporting ---
                        if progress_callback and self.perf:
                            p_stats = self.perf.get_progress()
                            base_prog = (current_chunk_idx_disp - 1) / total_chunks_disp
                            chunk_prog = p_stats.get('percent', 0) / 100.0
                            overall = min(1.0, base_prog + (chunk_prog / total_chunks_disp))
                            status = f"Chunk {current_chunk_idx_disp}/{total_chunks_disp}: {p_stats['percent']:.1f}% (FPS:{p_stats['fps']:.1f}|ETA:{p_stats['eta']})"
                            try: progress_callback(overall, status)
                            except Exception as cb_err: print(f"Warning: Gradio progress_callback error: {cb_err}")
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0):
                             prog_stats_console = self.perf.get_progress()
                             active_trk_count = self.tracker.get_active_vehicle_count() if hasattr(self.tracker, 'get_active_vehicle_count') else 'N/A'
                             print(f"\rChunk {current_chunk_idx_disp}/{total_chunks_disp} Prog:{prog_stats_console['percent']:.1f}%|FPS:{prog_stats_console['fps']:.1f}|Active:{active_trk_count}|ETA:{prog_stats_console['eta']} ", end="")
                             last_progress_print_time = current_mono_time

                        # --- Clear Batch Accumulators ---
                        frames_batch_gpu_tensors.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                        # Explicitly delete large tensors from this batch scope
                        del batch_argb_for_viz_and_model, yolo_model_input, yolo_results, batch_log_entries, batch_viz_data_for_gpu
                        if _viz_enabled_this_run and self.writer_instance is not None:
                            if 'visualized_batch_argb_float' in locals() and visualized_batch_argb_float is not None:
                                del visualized_batch_argb_float
                            if 'frame_to_encode_argb_uint8' in locals():
                                del frame_to_encode_argb_uint8
                        cleanup_memory() # Call cleanup after deleting local batch vars

                    # Mark the item from the reader queue as processed
                    if self.frame_read_queue: self.frame_read_queue.task_done()

                except queue.Empty:
                    # Check if the reader thread is still alive
                    if self.reader_thread and not self.reader_thread.is_alive() and self.frame_read_queue.empty():
                        print(f"\n_process_single_video_file: Reader thread finished and queue empty. Exiting loop.")
                        break
                    else:
                        # Timeout occurred, but reader might still be working or just slow.
                        # print(f"\n_process_single_video_file: Frame reader queue timeout. Reader alive: {self.reader_thread.is_alive() if self.reader_thread else 'N/A'}. Queue empty: {self.frame_read_queue.empty()}. Continuing wait.")
                        continue # Continue waiting for more frames
                except Exception as e_loop:
                    print(f"\n_process_single_video_file: ERROR during processing loop: {e_loop}")
                    import traceback; traceback.print_exc()
                    if self.stop_event: self.stop_event.set(); # Signal other threads
                    break # Exit loop on error
            # --- End Main Processing Loop ---

            # --- Process Final Batch (if any frames remain) ---
            if frames_batch_gpu_tensors:
                processed_frame_indices_log.extend(frame_numbers_batch_abs)
                print(f"\n_process_single_video_file: Processing final batch of {len(frames_batch_gpu_tensors)} frames...")
                # --- THIS IS A COPY OF THE BATCH PROCESSING LOGIC ---
                batch_start_mono = time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing')
                batch_argb_for_viz_and_model = self._preprocess_batch(frames_batch_gpu_tensors)
                if self.perf: self.perf.end_timer('preprocessing')

                batch_log_entries = []
                batch_viz_data_for_gpu = []
                num_processed_in_batch = len(frames_batch_gpu_tensors)

                if batch_argb_for_viz_and_model is not None and batch_argb_for_viz_and_model.nelement() > 0:
                    yolo_model_input = batch_argb_for_viz_and_model[:, 1:4, :, :] # RGB for model
                    # --- Inference ---
                    print(f"Debug VideoProcessor: Shape passed to model: {yolo_model_input.shape}, Dtype: {yolo_model_input.dtype}")
                    yolo_results = self.model(yolo_model_input, verbose=False)
                    # --- Post-processing ---
                    batch_log_entries, batch_viz_data_for_gpu = self._process_inference_results(
                        yolo_results, yolo_model_input, frame_numbers_batch_abs, frame_times_batch_abs, start_datetime_chunk
                    )
                    all_detections_structured_log.extend(batch_log_entries)
                    # --- GPU Visualization & Writing ---
                    if _viz_enabled_this_run and self.writer_instance is not None:
                        if self.perf: self.perf.start_timer('drawing_gpu')
                        visualized_batch_argb_float = add_gpu_overlays(
                            batch_argb_for_viz_and_model, batch_viz_data_for_gpu, self.zone_tracker.zones
                        )
                        if self.perf: self.perf.end_timer('drawing_gpu')
                        if visualized_batch_argb_float is not None:
                            for frame_tensor_argb_float in visualized_batch_argb_float:
                                frame_to_encode_argb_uint8 = (frame_tensor_argb_float.clamp(0,1) * 255.0).byte()
                                self.writer_instance.put(frame_to_encode_argb_uint8)

                frames_processed_count += num_processed_in_batch # Count before clearing

                # <<< Print Summary of Processed Frames >>>
                print("\n--- Processed Frame Index Summary ---")
                if processed_frame_indices_log:
                    processed_frame_indices_log.sort()
                    count = len(processed_frame_indices_log)
                    min_f = processed_frame_indices_log[0]
                    max_f = processed_frame_indices_log[-1]
                    print(f"Total frames processed: {count}")
                    print(f"Min frame index: {min_f}")
                    print(f"Max frame index: {max_f}")
                    # Print first and last few for quick check
                    print(f"First 10 indices: {processed_frame_indices_log[:10]}")
                    print(f"Last 10 indices: {processed_frame_indices_log[-10:]}")
                    # Check for gaps (simple check)
                    gaps = [(processed_frame_indices_log[i] + 1) for i in range(count - 1) if processed_frame_indices_log[i+1] != processed_frame_indices_log[i] + 1]
                    if gaps:
                        print(f"Warning: Potential gaps found after frames: {gaps[:20]}...") # Print first 20 gaps
                    else:
                        print("No gaps found in processed frame indices (sequential).")
                else:
                    print("No frames were processed.")
                print("--- End Processed Frame Index Summary ---")
                
                # Mark tasks done for the final batch items from read_queue
                num_tasks_to_done_final = len(frames_batch_gpu_tensors)
                if self.frame_read_queue:
                    for _ in range(num_tasks_to_done_final):
                        try: self.frame_read_queue.task_done()
                        except ValueError: break 

                # Clear final batch lists and delete tensors
                frames_batch_gpu_tensors.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                del batch_argb_for_viz_and_model, yolo_model_input, yolo_results, batch_log_entries, batch_viz_data_for_gpu
                if _viz_enabled_this_run and self.writer_instance is not None:
                    if 'visualized_batch_argb_float' in locals() and visualized_batch_argb_float is not None:
                        del visualized_batch_argb_float
                    if 'frame_to_encode_argb_uint8' in locals():
                        del frame_to_encode_argb_uint8
                cleanup_memory()
                print(f"Finished processing final batch. Total frames processed by inference: {frames_processed_count}")

            # --- Finalize Stage (Forced Exits, Thread Joins, Reporting) ---
            print(f"_process_single_video_file: Finalizing for {base_video_name}...")
            if self.tracker and hasattr(self.tracker, 'active_vehicles') and self.tracker.active_vehicles:
                 print(f"Forcing exit for {len(self.tracker.active_vehicles)} remaining active vehicles...")
                 last_ts_in_chunk = start_datetime_chunk + timedelta(seconds=frame_times_batch_abs[-1]) if frame_times_batch_abs else start_datetime_chunk
                 for v_id_force in list(self.tracker.active_vehicles.keys()):
                     if self.tracker.force_exit_vehicle(v_id_force, last_ts_in_chunk):
                         if self.perf: self.perf.record_vehicle_exit('forced_exit')
                         p_data = next((p for p in reversed(getattr(self.tracker, 'completed_paths', [])) if p['id']==v_id_force and p['status']=='forced_exit'), None)
                         if p_data:
                              all_detections_structured_log.append({'timestamp_dt': last_ts_in_chunk, 'vehicle_id': v_id_force, 'vehicle_type': p_data.get('type', 'Unknown'), 'event_type':'FORCED_EXIT', 'direction_from': p_data.get('entry_direction','UNKNOWN'), 'direction_to': 'FORCED', 'status':'forced_exit', 'time_in_intersection': p_data.get('time_in_intersection', 'N/A'), 'frame_number': 'END'})

            # Signal threads to stop (reader might have already finished)
            print(f"_process_single_video_file: Signaling stop_event for {base_video_name}...");
            if self.stop_event: self.stop_event.set()

            # Signal EOS to writer if it was initialized and running
            if self.writer_instance is not None:
                 print(f"Finalize: Sending EOS (None) to writer queue for {base_video_name}...");
                 self.writer_instance.put(None) # EOS for writer

            # Join reader thread
            if self.reader_thread and self.reader_thread.is_alive():
                print(f"Finalize: Joining reader thread for {base_video_name}...")
                self.reader_thread.join(timeout=10)
                if self.reader_thread.is_alive(): print(f"Warning: Reader thread join timed out for {base_video_name}.")
                else: print(f"Reader thread joined for {base_video_name}.")
            
            # Stop and join writer thread (stop includes muxing)
            mux_success_this_file = True # Assume success if no writer
            if self.writer_instance is not None: # Check if writer object exists
                 print(f"Finalize: Stopping writer (includes muxing) for {base_video_name}...")
                 mux_success_this_file = self.writer_instance.stop() # This blocks & runs muxing
                 if mux_success_this_file and hasattr(self.writer_instance, 'output_path'):
                     self.final_output_path_current_file = self.writer_instance.output_path
                 else:
                     self.final_output_path_current_file = None
                     print(f"Warning: Muxing failed or writer stopped improperly for {base_video_name}.")
            elif _viz_enabled_this_run:
                 print(f"Finalize: Writer was not initialized, skipping writer stop/muxing for {base_video_name}.")

            # --- Performance Summary ---
            total_time_taken_file=0.0; fps_file=0.0; completed_paths_count_valid_file=0
            if self.perf:
                self.perf.end_processing()
                self.perf.print_summary()
                total_time_taken_file=self.perf.total_time
                fps_file=frames_processed_count/total_time_taken_file if total_time_taken_file>0 else 0
            
            final_completed_paths_for_report = []
            if self.tracker and hasattr(self.tracker, 'get_completed_paths'):
                 final_completed_paths_for_report = self.tracker.get_completed_paths()
            elif self.tracker and hasattr(self.tracker, 'completed_paths'):
                 final_completed_paths_for_report = self.tracker.completed_paths

            # Calculate valid completed paths for summary
            for path_rep in final_completed_paths_for_report:
                status = path_rep.get('status')
                exit_dir = path_rep.get('exit_direction')
                if status == 'exited' and path_rep.get('entry_direction') and exit_dir and \
                   exit_dir not in ['TIMEOUT', 'FORCED', 'UNKNOWN', None]:
                    completed_paths_count_valid_file +=1

            # --- Create report for THIS CHUNK/FILE ---
            print(f"Generating report for {base_video_name}...")
            excel_file_path = create_excel_report(
                completed_paths_data=final_completed_paths_for_report,
                start_datetime=start_datetime_chunk,
                primary_direction=primary_direction,
                video_path=video_path,
                output_path_override=output_path_override
            )

            # --- Prepare Result Message ---
            result_msg_this_file = f"✅ Processing completed for {base_video_name}.\n"
            if _viz_enabled_this_run:
                 if self.final_output_path_current_file and mux_success_this_file:
                     result_msg_this_file += f"Output video segment: '{self.final_output_path_current_file}'.\n"
                 elif not mux_success_this_file:
                     result_msg_this_file += f"Output video segment generation failed (muxing error).\n"
                 else: # Writer wasn't initialized
                      result_msg_this_file += f"Video output writing was disabled or failed during initialization.\n"
            if excel_file_path:
                result_msg_this_file += f"Report segment: {excel_file_path}\n"
            else:
                result_msg_this_file += f"Report segment generation failed for {base_video_name}.\n"
            # Append Stats
            result_msg_this_file += f"--- Summary Stats for {base_video_name} ---\n"
            result_msg_this_file += f"STAT_FRAMES_PROCESSED={frames_processed_count}\n"
            result_msg_this_file += f"STAT_TIME_SECONDS={total_time_taken_file:.2f}\n"
            result_msg_this_file += f"STAT_FPS={fps_file:.2f}\n"
            result_msg_this_file += f"STAT_COMPLETED_PATHS={completed_paths_count_valid_file}\n" # Using valid count
            result_msg_this_file += f"--- End Stats ---"

            print(f"_process_single_video_file: Finished successfully for {base_video_name}.")
            return result_msg_this_file

        except Exception as e_outer:
            # Catch errors occurring anywhere in the process for this file
            print(f"\nFATAL ERROR during _process_single_video_file for {base_video_name}: {e_outer}")
            import traceback; traceback.print_exc()
            # Ensure threads are signalled to stop on error
            if hasattr(self, 'stop_event') and self.stop_event and not self.stop_event.is_set():
                print("Signalling stop event due to outer exception...")
                self.stop_event.set()
            # Attempt to stop writer gracefully if it exists
            if hasattr(self, 'writer_instance') and self.writer_instance:
                try:
                     print("Attempting to stop writer after outer exception...")
                     self.writer_instance.stop(cleanup_raw_file=True) # Attempt cleanup
                except Exception as e_stop:
                     print(f"Exception while trying to stop writer during error handling: {e_stop}")
            return f"❌ Error processing {base_video_name}: {e_outer}"
        finally:
            # --- Cleanup per-file resources ---
            print(f"_process_single_video_file: Entering finally block for {base_video_name}...")
            # Ensure reader thread is joined
            if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive():
                print(f"Finally: Joining reader thread for {base_video_name}...")
                self.reader_thread.join(timeout=5)
                if self.reader_thread.is_alive(): print(f"Warning: Reader thread join timed out in finally block for {base_video_name}.")
            
            # Writer thread should have been joined by writer_instance.stop() if called successfully
            # If stop wasn't called (e.g., error before writer init), writer_thread might not exist or be relevant

            # Clear instance variables to free resources and prevent state leakage
            self.tracker=None
            self.zone_tracker=None
            self.perf=None
            self.frame_read_queue=None
            self.reader_thread=None
            self.writer_instance=None # Ensure writer is cleared
            self.reader_instance=None # Ensure reader is cleared
            self.stop_event=None
            self.reader_dimensions_ready_event = None
            self.final_output_path_current_file = None
            self.detection_zones_polygons = None
            
            cleanup_memory() # Final cleanup attempt
            print(f"Finished resource cleanup for {base_video_name}")

# --- End of VideoProcessor class ---