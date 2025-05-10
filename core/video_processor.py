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
import kornia.color # Keep this import
import sys

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
            from visualization import overlay as visual_overlay # Fallback for CPU drawing if needed
    else:
        # Reader failed to import, so GPU viz is impossible
        print("NvidiaFrameReader failed to import, disabling GPU Visualization.")
        GPU_VIZ_ENABLED = False
        from visualization import overlay as visual_overlay # Fallback for CPU drawing if needed
else:
    print("ENABLE_VISUALIZATION is False. No video output will be generated.")
    from visualization import overlay as visual_overlay # Still import for CPU-based status overlay if pipelines use it

USE_CPU_READER_FALLBACK = False # Set to True if you want cv2.VideoCapture as fallback
if NvidiaFrameReader is None and not USE_CPU_READER_FALLBACK:
     print("CRITICAL WARNING: NvidiaFrameReader failed to import and CPU fallback is disabled. GPU-accelerated reading will not be available.")
elif NvidiaFrameReader is None and USE_CPU_READER_FALLBACK:
     print("Using CPU Frame Reader (cv2.VideoCapture) as fallback.")


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
        self.gradio_polygons = {}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.model_names = {}

        self.tracker_type_internal = TRACKER_TYPE
        self.tracker_lock = threading.Lock()

        # Attributes managed per file/chunk
        self.tracker = None 
        self.zone_tracker = None 
        self.perf = None 
        self.detection_zones_polygons = None
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.frame_read_queue = None
        self.video_write_queue = None 
        self.stop_event = None
        self.reader_thread = None
        self.writer_instance = None 
        self.writer_thread = None   
        self.last_cleanup_time = 0 
        self.final_output_path_current_file = None 
        self.reader_instance = None 
        self.reader_dimensions_ready_event = None 

        self.last_processed_frame_time_abs_in_loop = None 
        self._batch_processed_count_for_debug = 0 
        self.frames_batch_input_list_for_cpu_draw = [] # Added for CPU drawing path

        # CUDA Streams
        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0
        
        try:
            print("VideoProcessor: Initializing Model...")
            self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError("Model loading failed during init.")
            self.model_names = getattr(self.model, 'names', {})
            if not self.model_names: print("VideoProcessor: Warning - Could not get class names from loaded model.")
            print("VideoProcessor: Model loaded successfully in __init__.")
        except Exception as e:
            print(f"FATAL: Error loading model during VideoProcessor init: {e}")
            self.model = None

    def _draw_polygons_for_gradio(self, frame_to_draw_on):
        if frame_to_draw_on is None: return None
        img_copy = frame_to_draw_on.copy()
        colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} # BGR
        default_color = (255, 255, 255); temp_color = (0, 255, 255)
        for direction, points in self.gradio_polygons.items():
            if points and len(points) >= 3:
                pts_np = np.array(points, dtype=np.int32); color = colors.get(direction, default_color)
                cv2.polylines(img_copy, [pts_np], isClosed=True, color=color, thickness=2)
                try:
                    if len(pts_np) > 0: M = cv2.moments(pts_np)
                    if M["m00"] != 0: cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"]); cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2:
                last_pt = self.gradio_temp_points[-1]; first_pt = self.gradio_temp_points[0]
                hint_color = tuple(c // 2 for c in current_temp_color); cv2.line(img_copy, last_pt, first_pt, hint_color, 1, cv2.LINE_AA)
        self.last_drawn_polygons_frame = img_copy; return img_copy

    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive': return None, "Zone definition mode is 'hardcoded'. Drawing disabled."
        self.reset_gradio_polygons()
        if video_path is None: return None, "Please upload a video."
        try:
            cap=cv2.VideoCapture(str(video_path)); ret,frame=cap.read(); cap.release()
            if ret: frame=cv2.resize(frame,self.frame_shape); self.original_frame_for_gradio=frame.copy(); drawn_frame=self._draw_polygons_for_gradio(self.original_frame_for_gradio); return drawn_frame, "Video loaded. Select direction and click image to define zone vertices."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"

    def set_gradio_polygon_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        direction_lower = direction.lower()
        if direction_lower in self.gradio_polygons: del self.gradio_polygons[direction_lower]; print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower; self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status

    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1]); self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status

    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3: status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."; frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy(); status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None; self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status

    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else: removed_point = self.gradio_temp_points.pop(); status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status

    def reset_gradio_polygons(self):
        self.gradio_polygons={}; self.current_gradio_direction=None; self.gradio_temp_points=[]
        status="All polygons cleared."; frame_to_show=self._draw_polygons_for_gradio(self.original_frame_for_gradio) if self.original_frame_for_gradio is not None else None
        self.last_drawn_polygons_frame = frame_to_show; return frame_to_show, status

    def _frame_reader_task(self, video_path, original_fps): # CPU Fallback Reader
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened(): print(f"ERROR: CPU Reader failed to open video: {video_path}", flush=True); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps)); frame_count = 0; frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read();
                if not ret: break
                if frame_count % frame_interval == 0:
                    frame_time_seconds = frame_count / original_fps
                    try:
                        self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                        frames_yielded += 1
                    except queue.Full:
                        debug_print(f"CPU Reader queue full for {video_path}, waiting...")
                        time.sleep(0.1);
                        if self.stop_event.is_set(): break
                        try: 
                            self.frame_read_queue.put((frame, frame_count, frame_time_seconds), block=True, timeout=5.0)
                            frames_yielded += 1
                        except queue.Full: print(f"ERROR: CPU Reader queue persistently full for {video_path}.", flush=True); break
                    except Exception as e: debug_print(f"CPU Reader queue error for {video_path}: {e}"); break
                frame_count += 1
            debug_print(f"CPU Reader finished for {video_path}. Read {frame_count}, yielded ~{frames_yielded}.")
        except Exception as e: print(f"ERROR in CPU reader thread for {video_path}: {e}", flush=True)
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, timeout=1.0) 
                except queue.Full: pass

    def _frame_writer_task(self, output_path): # CPU Fallback Writer
        out = None; writer_initialized = False; frames_written = 0
        actual_output_path = output_path
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI.", flush=True);
                avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID');
                out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened(): actual_output_path = avi_path; self.final_output_path_current_file = avi_path 
                else: print(f"ERROR: CPU Writer failed to open for both MP4 and AVI for {output_path}.", flush=True); self.final_output_path_current_file = "ERROR_WRITER_FAILED"; return 
            writer_initialized = True; print(f"CPU Writer initialized for: {actual_output_path}", flush=True)
            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0)
                except queue.Empty:
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue 
                if frame_data is None: break 
                frame = frame_data
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]: frame = cv2.resize(frame, self.frame_shape)
                    out.write(frame); frames_written += 1
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e:
            print(f"ERROR in CPU writer thread for {output_path}: {e}", flush=True)
            if hasattr(self, 'final_output_path_current_file') and self.final_output_path_current_file != "ERROR_WRITER_FAILED": self.final_output_path_current_file = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            if out and writer_initialized: out.release(); debug_print(f"CPU Writer released. Wrote {frames_written} frames to {actual_output_path}")
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass
                
    @torch.no_grad()
    def _preprocess_batch(self, current_batch_of_gpu_tensors_from_reader):
        log_prefix = getattr(self.reader_instance, 'thread_name_prefix', 'Preproc') if hasattr(self, 'reader_instance') and self.reader_instance else 'Preproc'
        if not current_batch_of_gpu_tensors_from_reader:
            print(f"{log_prefix}: Received empty input tensor list for preprocessing.", flush=True)
            return None
        processed_argb_tensors_for_batch = []
        target_h_model = getattr(self, 'model_input_size_config', 416)
        target_w_model = getattr(self, 'model_input_size_config', 416)
        if not (hasattr(self, 'reader_instance') and self.reader_instance is not None and \
                hasattr(self.reader_instance, 'width') and self.reader_instance.width > 0 and \
                hasattr(self.reader_instance, 'height') and self.reader_instance.height > 0):
            print(f"{log_prefix}: CRITICAL - Reader instance or its dimensions not available for NV12 conversion. Cannot process batch.", flush=True)
            return None
        original_h = self.reader_instance.height
        original_w = self.reader_instance.width
        if original_h % 2 != 0 or original_w % 2 != 0:
            print(f"{log_prefix}: CRITICAL - Original video dimensions ({original_w}x{original_h}) not evenly divisible by 2. Kornia YUV420 conversion requires even dimensions. Batch processing aborted.", flush=True)
            return None

        for idx, raw_nv12_tensor in enumerate(current_batch_of_gpu_tensors_from_reader):
            if raw_nv12_tensor is None: print(f"{log_prefix}: Warning - Received None tensor at index {idx} in batch. Skipping.", flush=True); continue
            if str(raw_nv12_tensor.device) != str(self.device):
                try: raw_nv12_tensor = raw_nv12_tensor.to(self.device)
                except Exception as e_move: print(f"{log_prefix}: Error moving tensor for frame index {idx} to device {self.device}: {e_move}. Skipping frame.", flush=True); continue
            expected_h_total = int(original_h * 1.5)
            if raw_nv12_tensor.ndim != 2 or raw_nv12_tensor.shape != (expected_h_total, original_w): print(f"{log_prefix}: Unexpected raw_nv12_tensor shape: {raw_nv12_tensor.shape} for frame index {idx}. Expected 2D ({expected_h_total}, {original_w}). Skipping.", flush=True); continue
            if raw_nv12_tensor.dtype != torch.uint8: print(f"{log_prefix}: Warning (frame index {idx}) - NV12 tensor dtype is {raw_nv12_tensor.dtype}, expected torch.uint8. Skipping.", flush=True); continue
            try:
                y_plane = raw_nv12_tensor[:original_h, :]; uv_plane_interleaved = raw_nv12_tensor[original_h:, :]
                uv_plane_deinterleaved = uv_plane_interleaved.reshape(original_h // 2, original_w // 2, 2); uv_plane_permuted = uv_plane_deinterleaved.permute(2, 0, 1)
                y_plane_input = y_plane.float().unsqueeze(0).unsqueeze(0) / 255.0; uv_plane_permuted_float = uv_plane_permuted.float()
                uv_plane_scaled = (uv_plane_permuted_float / 255.0) - 0.5; uv_plane_input = uv_plane_scaled.unsqueeze(0)
            except Exception as e_reshape: print(f"{log_prefix}: Error separating/reshaping YUV planes for frame index {idx}: {e_reshape}. Skipping frame.", flush=True); continue
            try:
                frame_tensor_rgb_k = kornia.color.yuv420_to_rgb(y_plane_input, uv_plane_input)
                if frame_tensor_rgb_k is None: raise RuntimeError("kornia.color.yuv420_to_rgb returned None")
                frame_tensor_rgb = torch.clamp(frame_tensor_rgb_k, 0.0, 1.0).squeeze(0)
                if idx == 0 and DEBUG_MODE and (self._batch_processed_count_for_debug % 10 == 0):
                    debug_video_name_part = os.path.splitext(os.path.basename(getattr(self.reader_instance, 'video_path', 'unknown_video')))[0]
                    save_prefix = f"debug_{debug_video_name_part}_batch{self._batch_processed_count_for_debug}_frame{idx}"
                    try:
                        os.makedirs("debug_output", exist_ok=True)
                        save_path_y=os.path.join("debug_output",f"{save_prefix}_y_plane.png"); save_path_u=os.path.join("debug_output",f"{save_prefix}_u_plane.png"); save_path_v=os.path.join("debug_output",f"{save_prefix}_v_plane.png"); save_path_rgb=os.path.join("debug_output",f"{save_prefix}_kornia_rgb.png")
                        y_plane_np = y_plane.cpu().numpy(); cv2.imwrite(save_path_y, y_plane_np)
                        uv_plane_np = uv_plane_permuted.cpu().numpy(); cv2.imwrite(save_path_u, uv_plane_np[0]); cv2.imwrite(save_path_v, uv_plane_np[1])
                        temp_rgb_k_np = frame_tensor_rgb.permute(1,2,0).cpu().numpy(); print(f"{log_prefix}: Kornia Raw Output (Frame {idx} of Batch {self._batch_processed_count_for_debug}): Min={temp_rgb_k_np.min():.4f}, Max={temp_rgb_k_np.max():.4f}, Mean={temp_rgb_k_np.mean():.4f}", flush=True)
                        scaled_rgb_np_vis = np.clip(temp_rgb_k_np*255.0,0,255).astype(np.uint8); bgr_output_kornia=cv2.cvtColor(scaled_rgb_np_vis,cv2.COLOR_RGB2BGR); cv2.imwrite(save_path_rgb,bgr_output_kornia)
                        print(f"{log_prefix}: Saved Kornia debug images for {save_prefix} to 'debug_output/'", flush=True)
                    except Exception as e_save: print(f"{log_prefix}: Error saving debug images: {e_save}", flush=True)
            except Exception as e_conv: print(f"{log_prefix}: Error during kornia.color.yuv420_to_rgb for frame index {idx}: {e_conv}. Skipping.", flush=True); continue
            _c, current_h, current_w = frame_tensor_rgb.shape
            alpha_channel = torch.ones((1, current_h, current_w), dtype=frame_tensor_rgb.dtype, device=frame_tensor_rgb.device)
            frame_tensor_argb = torch.cat((alpha_channel, frame_tensor_rgb), dim=0)
            if current_h != target_h_model or current_w != target_w_model:
                try: frame_tensor_argb_resized = F.interpolate(frame_tensor_argb.unsqueeze(0), size=(target_h_model, target_w_model), mode='bilinear', align_corners=False).squeeze(0)
                except Exception as e_resize: print(f"{log_prefix}: Error during resize for frame index {idx}: {e_resize}. Skipping frame.", flush=True); continue
            else: frame_tensor_argb_resized = frame_tensor_argb
            processed_argb_tensors_for_batch.append(frame_tensor_argb_resized)
        
        self._batch_processed_count_for_debug += 1

        if not processed_argb_tensors_for_batch: print(f"{log_prefix}: No tensors were successfully processed in this batch.", flush=True); return None
        try: final_batch_tensor_argb = torch.stack(processed_argb_tensors_for_batch)
        except Exception as e_stack: print(f"{log_prefix}: Error stacking processed tensors into batch: {e_stack}", flush=True); return None
        return final_batch_tensor_argb
    
    def _process_inference_results(self, results_yolo, input_batch_tensor_for_shape_ref, frame_numbers, frame_times, start_datetime_chunk):
        batch_size_actual = len(results_yolo)
        batch_structured_log_data = [[] for _ in range(batch_size_actual)]
        batch_viz_output_list = [None] * batch_size_actual 

        def process_single_frame_logic(idx_in_batch):
            frame_number_abs = frame_numbers[idx_in_batch]
            print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - START processing.", flush=True)
            result = results_yolo[idx_in_batch]
            frame_time_sec_abs = frame_times[idx_in_batch]
            current_timestamp_abs = start_datetime_chunk + timedelta(seconds=frame_time_sec_abs)
            frame_log_entries = []; 
            single_frame_viz_output = [] if GPU_VIZ_ENABLED else None 

            output_frame_cpu = None 
            if ENABLE_VISUALIZATION and not GPU_VIZ_ENABLED:
                if hasattr(self, 'frames_batch_input_list_for_cpu_draw') and \
                   idx_in_batch < len(self.frames_batch_input_list_for_cpu_draw) and \
                   isinstance(self.frames_batch_input_list_for_cpu_draw[idx_in_batch], np.ndarray):
                    original_numpy_frame = self.frames_batch_input_list_for_cpu_draw[idx_in_batch].copy()
                    output_frame_cpu = cv2.resize(original_numpy_frame, self.frame_shape)
                else:
                    print(f"WARN: Original numpy frame for CPU viz not found for frame {frame_number_abs}. Using black frame.", flush=True)
                    output_frame_cpu = np.zeros((self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8)

            with self.tracker_lock:
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Acquired lock.", flush=True)
                if not self.tracker or not self.zone_tracker:
                    print(f"Error: Tracker or ZoneTracker not available for frame {frame_number_abs} (idx {idx_in_batch})", flush=True)
                    print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Releasing lock (early exit).", flush=True)
                    return [], single_frame_viz_output 

                detections_for_tracker = []
                tensor_h,tensor_w=input_batch_tensor_for_shape_ref.shape[2],input_batch_tensor_for_shape_ref.shape[3] 
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls_id=int(box.cls[0]); v_type_name=self.model_names.get(cls_id,f"CLS_{cls_id}")
                        x1_model,y1_model,x2_model,y2_model=map(int,box.xyxy[0].cpu().numpy())
                        x1_model,y1_model=max(0,x1_model),max(0,y1_model)
                        x2_model,y2_model=min(tensor_w-1,x2_model),min(tensor_h-1,y2_model)
                        center_pt_model=((x1_model+x2_model)//2,(y1_model+y2_model)//2)
                        bottom_center_pt_model=((x1_model+x2_model)//2,y2_model)
                        box_coords_to_store = (x1_model,y1_model,x2_model,y2_model)
                        center_pt_to_store = center_pt_model
                        bottom_center_for_zone_check = bottom_center_pt_model
                        detections_for_tracker.append({
                            'box_coords': box_coords_to_store, 
                            'center': center_pt_to_store,     
                            'bottom_center': bottom_center_for_zone_check, 
                            'type':v_type_name,
                            'confidence':float(box.conf[0])
                        })
                
                ids_seen_this_frame_custom=set(); processed_tracks_this_frame_map={}
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Before find_best_match loop.", flush=True)
                if self.tracker_type_internal == 'Custom':
                    matched_ids_map={}; unmatched_detections_indices=[]
                    for det_idx,det_info in enumerate(detections_for_tracker):
                        matched_id=self.tracker.find_best_match(det_info['center'],frame_time_sec_abs)
                        if matched_id: matched_ids_map[det_idx]=matched_id
                        else: unmatched_detections_indices.append(det_idx)
                    print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - After find_best_match loop (found {len(matched_ids_map)} matches, {len(unmatched_detections_indices)} new).", flush=True)
                    for det_idx,v_id in matched_ids_map.items():
                        det_info=detections_for_tracker[det_idx]; v_id,consistent_type=self.tracker.update_track(v_id,det_info['center'],frame_time_sec_abs,det_info['type'])
                        processed_tracks_this_frame_map[v_id]={'box':det_info['box_coords'],'type':consistent_type,'id':v_id,'center_for_zone':det_info['bottom_center'],'event':None,'status':'detected'}
                        ids_seen_this_frame_custom.add(v_id)
                    for det_idx in unmatched_detections_indices:
                        det_info=detections_for_tracker[det_idx]; v_id=self.tracker._get_next_id(); v_id,consistent_type=self.tracker.update_track(v_id,det_info['center'],frame_time_sec_abs,det_info['type'])
                        processed_tracks_this_frame_map[v_id]={'box':det_info['box_coords'],'type':consistent_type,'id':v_id,'center_for_zone':det_info['bottom_center'],'event':None,'status':'new'}
                        ids_seen_this_frame_custom.add(v_id)
                
                ids_to_remove_this_frame=set(); active_zone_keys=list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Before zone transition loop.", flush=True)
                for v_id, track_data in processed_tracks_this_frame_map.items():
                    current_bbox_for_zone_model_coords=track_data['box'] 
                    prev_pos_model_coords=None 
                    if v_id in self.tracker.tracking_history and len(self.tracker.tracking_history[v_id]) > 1: prev_pos_model_coords=self.tracker.tracking_history[v_id][-2]
                    
                    if prev_pos_model_coords and self.zone_tracker and self.zone_tracker.zones:
                        out_h_zone, out_w_zone = self.frame_shape[1], self.frame_shape[0]
                        prev_x_scaled_zone = int(prev_pos_model_coords[0] * out_w_zone / tensor_w)
                        prev_y_scaled_zone = int(prev_pos_model_coords[1] * out_h_zone / tensor_h)
                        prev_pos_scaled_for_zone = (prev_x_scaled_zone, prev_y_scaled_zone)
                        x1m, y1m, x2m, y2m = current_bbox_for_zone_model_coords
                        x1s_zone = int(x1m * out_w_zone / tensor_w); y1s_zone = int(y1m * out_h_zone / tensor_h)
                        x2s_zone = int(x2m * out_w_zone / tensor_w); y2s_zone = int(y2m * out_h_zone / tensor_h)
                        current_bbox_scaled_for_zone = (x1s_zone, y1s_zone, x2s_zone, y2s_zone)
                        
                        if self.perf: self.perf.start_timer('zone_checking')
                        event_type,event_dir=self.zone_tracker.check_zone_transition(prev_pos_scaled_for_zone, current_bbox_scaled_for_zone, v_id, frame_time_sec_abs)
                        if self.perf: self.perf.end_timer('zone_checking')
                        
                        v_state=self.tracker.active_vehicles.get(v_id); consistent_type=track_data['type']
                        if event_type=="ENTRY":
                            if v_state and v_state.get('status')=='active':
                                stored_entry_dir = v_state.get('entry_direction')
                                if stored_entry_dir and stored_entry_dir != event_dir and is_valid_movement(stored_entry_dir,event_dir,active_zone_keys):
                                    final_type_inf=self.tracker.get_consistent_type(v_id,consistent_type); success_inf,t_in_int_inf=self.tracker.register_exit(v_id,event_dir,current_timestamp_abs,track_data['center_for_zone'])
                                    if success_inf:
                                        if self.perf:self.perf.record_vehicle_exit('exited',t_in_int_inf)
                                        frame_log_entries.append({'timestamp_dt':current_timestamp_abs,'vehicle_id':v_id,'vehicle_type':final_type_inf,'event_type':'EXIT','direction_from':stored_entry_dir,'direction_to':event_dir,'status':'exited_inferred','time_in_intersection':t_in_int_inf,'frame_number':frame_number_abs})
                                        ids_to_remove_this_frame.add(v_id); processed_tracks_this_frame_map[v_id]['event']='EXIT'
                                    if self.tracker.register_entry(v_id,event_dir,current_timestamp_abs,track_data['center_for_zone'],consistent_type):
                                        if self.perf:self.perf.record_vehicle_entry()
                                        frame_log_entries.append({'timestamp_dt':current_timestamp_abs,'vehicle_id':v_id,'vehicle_type':consistent_type,'event_type':'ENTRY','direction_from':event_dir,'direction_to':None,'status':'entry','frame_number':frame_number_abs})
                                        processed_tracks_this_frame_map[v_id]['event']='ENTRY'; processed_tracks_this_frame_map[v_id]['status']='active'
                            if not (v_state and v_state.get('status')=='active'):
                                if self.tracker.register_entry(v_id,event_dir,current_timestamp_abs,track_data['center_for_zone'],consistent_type):
                                    if self.perf:self.perf.record_vehicle_entry()
                                    frame_log_entries.append({'timestamp_dt':current_timestamp_abs,'vehicle_id':v_id,'vehicle_type':consistent_type,'event_type':'ENTRY','direction_from':event_dir,'direction_to':None,'status':'entry','frame_number':frame_number_abs})
                                    processed_tracks_this_frame_map[v_id]['event']='ENTRY'; processed_tracks_this_frame_map[v_id]['status']='active'
                        elif event_type=="EXIT":
                            if v_state and v_state.get('status')=='active':
                                entry_dir_trk = v_state.get('entry_direction')
                                if entry_dir_trk and is_valid_movement(entry_dir_trk,event_dir,active_zone_keys):
                                    final_exit_type=self.tracker.get_consistent_type(v_id,consistent_type); success_ex,t_in_int_ex=self.tracker.register_exit(v_id,event_dir,current_timestamp_abs,track_data['center_for_zone']) # Use center_for_zone for exit point
                                    if success_ex:
                                        if self.perf:self.perf.record_vehicle_exit('exited',t_in_int_ex)
                                        frame_log_entries.append({'timestamp_dt':current_timestamp_abs,'vehicle_id':v_id,'vehicle_type':final_exit_type,'event_type':'EXIT','direction_from':entry_dir_trk,'direction_to':event_dir,'status':'exit','time_in_intersection':t_in_int_ex,'frame_number':frame_number_abs})
                                        ids_to_remove_this_frame.add(v_id); processed_tracks_this_frame_map[v_id]['event']='EXIT'; processed_tracks_this_frame_map[v_id]['status']='exited'
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - After zone transition loop.", flush=True)
                
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Before timeout check.", flush=True)
                timed_out_ids=self.tracker.check_timeouts(current_timestamp_abs)
                for v_id_to in timed_out_ids:
                    ids_to_remove_this_frame.add(v_id_to);
                    if self.perf:self.perf.record_vehicle_exit('timed_out')
                    p_data=next((p for p in reversed(self.tracker.completed_paths) if p['id']==v_id_to and p['status']=='timed_out'),None)
                    if p_data: frame_log_entries.append({'timestamp_dt':current_timestamp_abs,'vehicle_id':v_id_to,'vehicle_type':p_data.get('type','UnknownType'),'event_type':'TIMEOUT','direction_from':p_data.get('entry_direction','UNKNOWN'),'direction_to':'TIMEOUT','status':'timeout','time_in_intersection':p_data.get('time_in_intersection','N/A'),'frame_number':frame_number_abs})
                    if v_id_to in processed_tracks_this_frame_map: processed_tracks_this_frame_map[v_id_to]['status']='timed_out'
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - After timeout check.", flush=True)

                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Before increment_misses.", flush=True)
                if self.tracker_type_internal == 'Custom': self.tracker.increment_misses(ids_seen_this_frame_custom)
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Before remove_vehicle_data loop.", flush=True)
                for v_id_rem in ids_to_remove_this_frame: 
                    self.tracker.remove_vehicle_data(v_id_rem) 
                    if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_rem)
                print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - After remove_vehicle_data loop.", flush=True)
                
                current_active_vehicles_state_snapshot = self.tracker.active_vehicles.copy()
            print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - Releasing lock.", flush=True)
            
            if ENABLE_VISUALIZATION:
                if GPU_VIZ_ENABLED: 
                    for v_id, data in processed_tracks_this_frame_map.items():
                        if v_id not in ids_to_remove_this_frame:
                            viz_status='detected'; vehicle_state_for_viz=current_active_vehicles_state_snapshot.get(v_id)
                            if vehicle_state_for_viz and vehicle_state_for_viz.get('status')=='active': viz_status='active'
                            elif data.get('event')=='EXIT': viz_status='exiting'
                            elif data.get('event')=='ENTRY' and not (vehicle_state_for_viz and vehicle_state_for_viz.get('status')=='active'): viz_status='entering'
                            single_frame_viz_output.append({'box':data['box'],'id':data['id'],'type':data['type'],'status':viz_status})
                elif output_frame_cpu is not None: 
                    if self.perf: self.perf.start_timer('drawing_cpu') 
                    if self.zone_tracker and self.zone_tracker.zones: visual_overlay.draw_zones(output_frame_cpu, self.zone_tracker)
                    for v_id_draw, data_draw in processed_tracks_this_frame_map.items():
                        model_h_viz, model_w_viz = input_batch_tensor_for_shape_ref.shape[2], input_batch_tensor_for_shape_ref.shape[3]
                        out_h_viz, out_w_viz = self.frame_shape[1], self.frame_shape[0]
                        x1_m, y1_m, x2_m, y2_m = data_draw['box'] 
                        x1_s = int(x1_m * out_w_viz / model_w_viz); y1_s = int(y1_m * out_h_viz / model_h_viz)
                        x2_s = int(x2_m * out_w_viz / model_w_viz); y2_s = int(y2_m * out_h_viz / model_h_viz)
                        scaled_box_for_cpu_draw = (x1_s, y1_s, x2_s, y2_s)
                        center_x_m, center_y_m = data_draw['center'] # This is the true center, model-relative
                        scaled_center_x_cpu = int(center_x_m * out_w_viz / model_w_viz)
                        scaled_center_y_cpu = int(center_y_m * out_h_viz / model_h_viz)
                        scaled_center_for_cpu_draw = (scaled_center_x_cpu, scaled_center_y_cpu)
                        c_type_draw = data_draw['type']
                        display_status_cpu = "detected"; entry_dir_draw_cpu = None; t_active_draw_cpu = None
                        v_state_cpu = current_active_vehicles_state_snapshot.get(v_id_draw)
                        if v_state_cpu and v_state_cpu.get('status') == 'active':
                            display_status_cpu='active'; entry_dir_draw_cpu=v_state_cpu.get('entry_direction'); t_active_draw_cpu=(current_timestamp_abs - v_state_cpu['entry_time']).total_seconds() if v_state_cpu.get('entry_time') else None
                        if data_draw.get('status') != 'timed_out': 
                            visual_overlay.draw_detection_box(output_frame_cpu, scaled_box_for_cpu_draw, v_id_draw, c_type_draw, status=display_status_cpu, entry_dir=entry_dir_draw_cpu, time_active=t_active_draw_cpu)
                        
                        trail_model_coords = self.tracker.get_tracking_trail(v_id_draw) # These are model-relative centers
                        trail_scaled_cpu = [(int(pt[0] * out_w_viz / model_w_viz), int(pt[1] * out_h_viz / model_h_viz)) for pt in trail_model_coords]
                        visual_overlay.draw_tracking_trail(output_frame_cpu, trail_scaled_cpu, v_id_draw)
                        
                        event_marker_cpu = data_draw.get('event');
                        if event_marker_cpu: visual_overlay.draw_event_marker(output_frame_cpu, scaled_center_for_cpu_draw, event_marker_cpu, v_id_draw[-4:]) # Use scaled center for marker
                    visual_overlay.add_status_overlay(output_frame_cpu, frame_number_abs, current_timestamp_abs, self.tracker)
                    if self.perf: self.perf.end_timer('drawing_cpu')
                    single_frame_viz_output = output_frame_cpu
            
            print(f"DEBUG_PIF: Frame {frame_number_abs} (batch_idx {idx_in_batch}) - END processing.", flush=True)
            return frame_log_entries, single_frame_viz_output


        if self.perf: self.perf.start_timer('detection_processing')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
             futures = {executor.submit(process_single_frame_logic, i): i for i in range(batch_size_actual)}
             for i_future, future in enumerate(concurrent.futures.as_completed(futures)): 
                 original_submission_idx = futures[future]
                 try:
                     frame_log, viz_output_for_frame = future.result()
                     batch_structured_log_data[original_submission_idx] = frame_log
                     batch_viz_output_list[original_submission_idx] = viz_output_for_frame
                 except Exception as exc:
                     print(f'\nError in post-processing frame {frame_numbers[original_submission_idx]} (batch idx {original_submission_idx}): {exc}', flush=True); import traceback; traceback.print_exc();
                     batch_structured_log_data[original_submission_idx] = []
                     batch_viz_output_list[original_submission_idx] = [] if GPU_VIZ_ENABLED else None
        if self.perf: self.perf.end_timer('detection_processing')
        
        flat_log_data = [item for sublist in batch_structured_log_data for item in sublist]
        return flat_log_data, batch_viz_output_list


    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        if not self.processing_lock.acquire(blocking=False):
            yield {'final_message': "Processing is already in progress."}
            return
        yield "Orchestrator: Initializing processing..."
        print(f"--- Orchestrator: Starting Video Processing ---", flush=True); print(f"Video: {video_path}", flush=True)
        final_outcome_message = "Processing did not complete as expected."
        if self.model is None:
            yield "Orchestrator: Loading model..."
            print("Orchestrator: Model is None, attempting to load...", flush=True)
            try:
                self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
                if self.model is None: raise RuntimeError("Model loading returned None in orchestrator.")
                self.model_names = getattr(self.model, 'names', {})
                yield "Orchestrator: Model loaded successfully."
            except Exception as e:
                error_msg = f"❌ FATAL: Error loading model: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}; return
        def parse_dt(date_str, time_str):
            try: dt_str=f"{date_str}{time_str[:6]}"; base_dt=datetime.strptime(dt_str,"%y%m%d%H%M%S"); return base_dt.replace(microsecond=int(time_str[6:].ljust(6,'0')))
            except Exception as e: print(f"Error parsing datetime: {date_str} {time_str} - {e}", flush=True); return None
        original_start_dt = parse_dt(start_date_str, start_time_str)
        if not original_start_dt:
            error_msg = "❌ Error: Invalid start date/time format."
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}; return
        yield "Orchestrator: Reading video properties..."
        duration_sec, original_fps_prop, total_frames_prop = get_video_properties(video_path)
        if duration_sec is None:
            error_msg = f"❌ Error: Cannot read video properties: {video_path}"
            if self.processing_lock.locked(): self.processing_lock.release()
            yield {'final_message': error_msg}; return
        should_chunk = ENABLE_AUTO_CHUNKING and duration_sec > (AUTO_CHUNK_THRESHOLD_MINUTES*60)
        chunk_files=[video_path]; chunk_duration_sec_cfg=AUTO_CHUNK_DURATION_MINUTES*60
        if should_chunk:
            status_msg_chunking=f"Orchestrator: Video duration ({duration_sec:.0f}s) triggers chunking (~{AUTO_CHUNK_DURATION_MINUTES} min chunks)."
            yield status_msg_chunking; print(status_msg_chunking, flush=True)
            try:
                yield "Orchestrator: Splitting video now... (this may take a while)"
                chunk_files=split_video_ffmpeg(video_path,CHUNK_TEMP_DIR,chunk_duration_sec_cfg,FFMPEG_PATH,progress_callback=None)
                if not chunk_files: raise RuntimeError("FFmpeg splitting yielded no files.")
                yield f"Orchestrator: Video split into {len(chunk_files)} chunks."
            except Exception as e:
                error_msg=f"❌ Error during video splitting: {e}"
                if self.processing_lock.locked(): self.processing_lock.release()
                yield {'final_message': error_msg}; return
            if os.path.exists(REPORT_TEMP_DIR): shutil.rmtree(REPORT_TEMP_DIR)
            os.makedirs(REPORT_TEMP_DIR,exist_ok=True)
        else: status_msg_single=f"Orchestrator: Processing as single file (Duration: {duration_sec:.0f}s)."; yield status_msg_single; print(status_msg_single, flush=True)
        all_chunk_processing_ok = True; num_chunks = len(chunk_files); processed_chunk_results = []
        for i, current_chunk_path in enumerate(chunk_files):
            chunk_start_offset=timedelta(seconds=i*chunk_duration_sec_cfg if should_chunk else 0); current_chunk_start_dt=original_start_dt+chunk_start_offset
            current_start_date_str_chunk=current_chunk_start_dt.strftime("%y%m%d"); current_start_time_str_chunk=current_chunk_start_dt.strftime("%H%M%S")+f"{current_chunk_start_dt.microsecond//1000:03d}"
            status_processing_chunk=f"Orchestrator: Starting segment {i+1}/{num_chunks}: {os.path.basename(current_chunk_path)}"
            yield status_processing_chunk; print("*"*60, flush=True); print(status_processing_chunk, flush=True)
            temp_report_override_path=None
            if should_chunk: base_chunk_name=os.path.splitext(os.path.basename(current_chunk_path))[0]; temp_report_override_path=os.path.join(REPORT_TEMP_DIR,f"report_{base_chunk_name}.xlsx")
            result_msg_part = self._process_single_video_file(video_path=current_chunk_path,start_date_str=current_start_date_str_chunk,start_time_str=current_start_time_str_chunk,primary_direction=primary_direction,output_path_override=temp_report_override_path,progress_callback=None,chunk_info=(i + 1, num_chunks))
            processed_chunk_results.append(result_msg_part)
            if "❌ Error" in result_msg_part or "FATAL" in result_msg_part:
                all_chunk_processing_ok=False; final_outcome_message=result_msg_part; error_stop_msg=f"Orchestrator: Error processing {os.path.basename(current_chunk_path)}. Halting."
                yield error_stop_msg; print(error_stop_msg, flush=True); break
            else: yield f"Orchestrator: Finished segment {i+1}/{num_chunks}."
        print("*"*60, flush=True)
        if all_chunk_processing_ok:
            if should_chunk:
                consolidation_start_msg="Orchestrator: Consolidating reports..."; yield consolidation_start_msg; print(consolidation_start_msg, flush=True)
                original_base=os.path.splitext(os.path.basename(video_path))[0]; final_consolidated_report_path=os.path.join(REPORT_OUTPUT_DIR,f"detection_logs_{original_base}_CONSOLIDATED.xlsx")
                os.makedirs(REPORT_OUTPUT_DIR,exist_ok=True)
                consolidated_path=consolidate_excel_reports(REPORT_TEMP_DIR,final_consolidated_report_path,original_start_dt.date())
                if consolidated_path:
                    full_summary="Chunk Processing Summaries:\n"+"\n\n".join(processed_chunk_results); full_summary+=f"\n\n✅ Processing completed.\nConsolidated report: {consolidated_path}"
                    final_outcome_message=full_summary; yield "Orchestrator: Report consolidation successful."; shutil.rmtree(REPORT_TEMP_DIR)
                else: final_outcome_message=f"⚠️ Processing finished, but report consolidation failed. Chunk reports in {REPORT_TEMP_DIR}."; yield "Orchestrator: Report consolidation failed."
            else:
                final_report_path_single=None; result_msg_part_single=processed_chunk_results[0] if processed_chunk_results else ""
                match = re.search(r"Report segment: (.+\.xlsx)",result_msg_part_single) or re.search(r"Excel report: (.+\.xlsx)",result_msg_part_single)
                if match: final_report_path_single=match.group(1)
                if final_report_path_single and os.path.exists(final_report_path_single): final_outcome_message=result_msg_part_single
                else: final_outcome_message=f"⚠️ Processing finished for single file, but report file not found. Message: {result_msg_part_single}"
        elif not final_outcome_message: final_outcome_message = "❌ Processing stopped due to errors in chunk processing."
        if should_chunk:
            cleanup_msg="Orchestrator: Cleaning up temporary video chunks..."; yield cleanup_msg; print(cleanup_msg, flush=True)
            try: shutil.rmtree(CHUNK_TEMP_DIR); yield "Orchestrator: Temporary video chunks cleaned up."
            except Exception as e: warn_cleanup_msg=f"Warning: Could not remove temp chunk dir: {e}"; yield warn_cleanup_msg; print(warn_cleanup_msg, flush=True)
        release_msg="Orchestrator: All processing stages finished. Releasing lock."; yield release_msg; print(release_msg, flush=True)
        if self.processing_lock.locked(): self.processing_lock.release()
        yield {'final_message': final_outcome_message}

    def _process_single_video_file(self, video_path, start_date_str, start_time_str, primary_direction, output_path_override=None, progress_callback=None, chunk_info=(1,1)):
        base_video_name = os.path.basename(video_path)
        print(f"_process_single_video_file: Initializing for {base_video_name}", flush=True)
        
        _viz_enabled_this_run = ENABLE_VISUALIZATION and GPU_VIZ_ENABLED and NvidiaFrameWriter is not None and add_gpu_overlays is not None
        _cpu_viz_enabled_this_run = ENABLE_VISUALIZATION and not _viz_enabled_this_run
        writer_initialized_this_run = False

        self.tracker=VehicleTracker(); 
        self.zone_tracker=None; self.perf=PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event=threading.Event(); 
        self.reader_dimensions_ready_event=threading.Event() 
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons=None; self.writer_instance=None; self.reader_instance=None; self.reader_thread=None
        self.final_output_path_current_file=None; self.frame_read_queue=None
        self._batch_processed_count_for_debug = 0 
        self.last_processed_frame_time_abs_in_loop = None
        self.frames_batch_input_list_for_cpu_draw = [] 

        try:
            try: dt_str=f"{start_date_str}{start_time_str[:6]}"; base_dt=datetime.strptime(dt_str,"%y%m%d%H%M%S"); start_datetime_chunk=base_dt.replace(microsecond=int(start_time_str[6:].ljust(6,'0')))
            except ValueError as e_dt: raise ValueError(f"Invalid start date/time format '{start_date_str} {start_time_str}': {e_dt}") from e_dt
            print(f"_process_single_video_file: Parsed start datetime: {start_datetime_chunk.isoformat()}", flush=True)
            
            print(f"_process_single_video_file: Setting up zones (mode: {self.line_mode})...", flush=True)
            if self.line_mode=='hardcoded':
                if not LINE_POINTS: raise ValueError("Config LINE_POINTS missing for hardcoded mode.")
                valid_polys={k:v for k,v in LINE_POINTS.items() if isinstance(v,list) and len(v)>=3}
                if not valid_polys: raise ValueError("No valid polygons found in config LINE_POINTS.")
                self.detection_zones_polygons={k:np.array(v,dtype=np.int32) for k,v in valid_polys.items()}
            elif self.line_mode=='interactive':
                if not self.gradio_polygons: raise ValueError("Gradio polygons not defined for interactive mode.")
                valid_polys={k:v for k,v in self.gradio_polygons.items() if v and len(v)>=3}
                if len(valid_polys)<2: raise ValueError("Need at least 2 valid Gradio zones defined.")
                self.detection_zones_polygons={k:np.array(v,dtype=np.int32) for k,v in valid_polys.items()}
            else: raise ValueError(f"Invalid LINE_MODE configured: {self.line_mode}")
            self.zone_tracker=ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("ZoneTracker initialization failed.")
            print(f"ZoneTracker initialized with {len(self.zone_tracker.zones)} zones: {list(self.zone_tracker.zones.keys())}", flush=True)
            
            if self.model is None: raise RuntimeError(f"Model was not loaded successfully prior to _process_single_video_file for {MODEL_PATH}")
            
            print(f"_process_single_video_file: Getting video properties for {base_video_name}...", flush=True)
            duration_sec_file,original_fps_file,total_frames_file=get_video_properties(video_path)
            if duration_sec_file is None: raise FileNotFoundError(f"Cannot open or read video properties for: {video_path}")
            if original_fps_file <= 0: print(f"Warning: Original FPS read as {original_fps_file}. Using default 30.0.", flush=True); original_fps_file=30.0
            print(f"Video Properties: Duration={duration_sec_file:.2f}s, OrigFPS={original_fps_file:.2f}, TotalFrames~={total_frames_file}", flush=True)
            
            self.frame_read_queue = queue.Queue(maxsize=self.batch_size * 4)
            use_nvidia_reader = NvidiaFrameReader is not None and not USE_CPU_READER_FALLBACK

            if use_nvidia_reader:
                print(f"_process_single_video_file: Setting up NvidiaFrameReader...", flush=True)
                self.reader_instance = NvidiaFrameReader(video_path=video_path,target_fps=self.target_fps,original_fps_hint=original_fps_file,frame_queue=self.frame_read_queue,stop_event=self.stop_event,dimensions_ready_event=self.reader_dimensions_ready_event,device_id_for_torch_output=0)
                self.reader_thread = threading.Thread(target=self.reader_instance.run, name=f"NvidiaReader-{base_video_name}", daemon=True)
            else: 
                print(f"_process_single_video_file: Setting up CPU Frame Reader (cv2.VideoCapture)...", flush=True)
                self.reader_thread = threading.Thread(target=self._frame_reader_task, args=(video_path, original_fps_file), name=f"CPUReader-{base_video_name}", daemon=True)
            self.reader_thread.start()
            print(f"{'NvidiaReader' if use_nvidia_reader else 'CPU Reader'} thread started.", flush=True)

            all_detections_structured_log=[]; frames_processed_count=0
            if self.perf: self.perf.start_processing(total_frames_file)
            frames_batch_input_list=[]; frame_numbers_batch_abs=[]; frame_times_batch_abs=[]
            last_progress_print_time=time.monotonic(); current_chunk_idx_disp,total_chunks_disp=chunk_info
            processed_frame_indices_log=[]; batch_count_log = 0
            
            print(f"_process_single_video_file: Entering main processing loop for {base_video_name}...", flush=True)
            while True:
                try:
                    frame_data=self.frame_read_queue.get(timeout=120)
                    if frame_data is None:
                        if self.frame_read_queue: self.frame_read_queue.task_done()
                        print(f"_process_single_video_file: Received EOS (None) from reader queue. Exiting processing loop.", flush=True); break
                    
                    current_frame_raw, frame_num_abs, frame_time_abs = frame_data
                    self.last_processed_frame_time_abs_in_loop = frame_time_abs

                    if current_frame_raw is None:
                        if self.frame_read_queue: self.frame_read_queue.task_done()
                        print(f"Warning: Received None frame/tensor from reader queue (Frame {frame_num_abs}). Skipping.", flush=True); continue
                    
                    if ENABLE_VISUALIZATION and not writer_initialized_this_run:
                        if _viz_enabled_this_run: 
                             print(f"_process_single_video_file: Checking if Nvidia writer needs initialization (Frame {frame_num_abs})...", flush=True)
                             print(f"Waiting for reader dimensions_ready_event...", flush=True)
                             event_was_set=self.reader_dimensions_ready_event.wait(timeout=10.0)
                             if event_was_set:
                                 reader_w=getattr(self.reader_instance,'width',0); reader_h=getattr(self.reader_instance,'height',0)
                                 if reader_w>0 and reader_h>0:
                                     print(f"Video dimensions ready: {reader_w}x{reader_h}. Initializing NvidiaFrameWriter...", flush=True)
                                     chunk_video_base=f"output_{os.path.splitext(base_video_name)[0]}"; final_muxed_path=os.path.join(VIDEO_OUTPUT_DIR,f"{chunk_video_base}{FINAL_VIDEO_EXTENSION}")
                                     os.makedirs(VIDEO_OUTPUT_DIR,exist_ok=True); writer_input_format="ARGB"
                                     self.writer_instance=NvidiaFrameWriter(output_path=final_muxed_path,width=reader_w,height=reader_h,fps=self.target_fps,encoder_codec=ENCODER_CODEC,bitrate=ENCODER_BITRATE,preset=ENCODER_PRESET,input_tensor_format_str=writer_input_format,temp_dir=CHUNK_TEMP_DIR,device_id=0)
                                     self.writer_instance.start()
                                     if hasattr(self.writer_instance,'encoder') and self.writer_instance.encoder is not None: writer_initialized_this_run=True; print("Nvidia Writer thread started successfully.", flush=True); self.final_output_path_current_file=final_muxed_path
                                     else: print("NvidiaWriter CRITICAL: _initialize_encoder failed. Writing disabled.", flush=True); self.writer_instance=None; writer_initialized_this_run=True
                                 else: print(f"Reader dimensions event set, but reader w/h still 0. Writing disabled.", flush=True); writer_initialized_this_run=True
                             else: print("Timeout waiting for reader dimensions. Writing disabled.", flush=True); writer_initialized_this_run=True
                        elif _cpu_viz_enabled_this_run: 
                            print(f"_process_single_video_file: Initializing CPU writer...", flush=True)
                            os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts_cpu=datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename_base_cpu=f"output_{os.path.splitext(os.path.basename(video_path))[0]}_{ts_cpu}"
                            output_path_cpu=os.path.join(VIDEO_OUTPUT_DIR, f"{output_filename_base_cpu}.mp4");
                            self.final_output_path_current_file = output_path_cpu 
                            self.video_write_queue = queue.Queue(maxsize=self.batch_size*4)
                            self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path_cpu,), daemon=True)
                            self.writer_thread.start()
                            print("CPU Video Writer thread started.", flush=True)
                            writer_initialized_this_run = True

                    frames_batch_input_list.append(current_frame_raw); frame_numbers_batch_abs.append(frame_num_abs); frame_times_batch_abs.append(frame_time_abs)
                    
                    if len(frames_batch_input_list)>=self.batch_size:
                        batch_count_log += 1; log_batch_id_str = f"Batch {batch_count_log} (first frame {frame_numbers_batch_abs[0] if frame_numbers_batch_abs else 'N/A'})"
                        processed_frame_indices_log.extend(frame_numbers_batch_abs); batch_start_mono=time.monotonic()
                        
                        if _cpu_viz_enabled_this_run: 
                            self.frames_batch_input_list_for_cpu_draw = [f.copy() for f in frames_batch_input_list if isinstance(f, np.ndarray)]


                        print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting preprocessing.", flush=True)
                        if self.perf: self.perf.start_timer('preprocessing')
                        
                        yolo_model_input = None
                        batch_argb_for_viz_and_model = None 

                        if use_nvidia_reader and isinstance(frames_batch_input_list[0], torch.Tensor):
                            batch_argb_for_viz_and_model=self._preprocess_batch(frames_batch_input_list)
                            if batch_argb_for_viz_and_model is not None and batch_argb_for_viz_and_model.nelement()>0:
                                yolo_model_input = batch_argb_for_viz_and_model[:, 1:4, :, :]
                        else: 
                            tensors_cpu = []
                            for cpu_frame_item in frames_batch_input_list:
                                if cpu_frame_item is None or not isinstance(cpu_frame_item, np.ndarray): continue
                                img_cpu=cv2.resize(cpu_frame_item,(self.model_input_size_config,self.model_input_size_config)); img_cpu=cv2.cvtColor(img_cpu,cv2.COLOR_BGR2RGB)
                                img_cpu=img_cpu.transpose(2,0,1); img_cpu=np.ascontiguousarray(img_cpu); tensors_cpu.append(img_cpu)
                            if tensors_cpu:
                                yolo_model_input =torch.from_numpy(np.stack(tensors_cpu)).to(self.device)
                                dtype_cpu=torch.float16 if (MIXED_PRECISION and self.device=='cuda') else torch.float32
                                yolo_model_input = yolo_model_input.to(dtype_cpu)/255.0
                        
                        if self.perf: self.perf.end_timer('preprocessing')
                        print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished preprocessing.", flush=True)
                        
                        batch_log_entries=[]; batch_viz_output_for_display =[]; num_processed_in_batch=len(frames_batch_input_list)
                        
                        if yolo_model_input is not None and yolo_model_input.nelement()>0:
                            if DEBUG_MODE:
                                TARGET_DEBUG_FRAME=100; pipeline_name="CurrentPipeline" 
                                if TARGET_DEBUG_FRAME in frame_numbers_batch_abs:
                                    try:
                                        batch_idx_debug=frame_numbers_batch_abs.index(TARGET_DEBUG_FRAME); tensor_to_debug=yolo_model_input[batch_idx_debug].detach().cpu()
                                        print(f"\n--- Input Tensor Stats for Frame {TARGET_DEBUG_FRAME} ({pipeline_name}) ---", flush=True)
                                        print(f"  Shape: {tensor_to_debug.shape}", flush=True); print(f"  Dtype: {tensor_to_debug.dtype}", flush=True); print(f"  Min: {tensor_to_debug.min().item():.6f}", flush=True); print(f"  Max: {tensor_to_debug.max().item():.6f}", flush=True); print(f"  Mean: {tensor_to_debug.mean().item():.6f}", flush=True); print(f"  Std: {tensor_to_debug.std().item():.6f}", flush=True)
                                        print("--- End Tensor Stats ---", flush=True)
                                    except Exception as e_stat: print(f"\nError getting tensor stats: {e_stat}", flush=True)
                            
                            stream=None
                            if self.device=='cuda' and self.streams: stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                            with torch.cuda.stream(stream) if stream else torch.no_grad():
                                with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                                    if self.perf and self.perf.start_event: self.perf.start_event.record(stream=stream)
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting model inference. Shape: {yolo_model_input.shape}, Dtype: {yolo_model_input.dtype}, Expected imgsz: {self.model_input_size_config}", flush=True)
                                    yolo_results=self.model(yolo_model_input,imgsz=self.model_input_size_config,verbose=False)
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished model inference.", flush=True)
                                    if self.perf and self.perf.end_event: self.perf.end_event.record(stream=stream)
                            if stream: stream.synchronize()
                            if self.perf and self.perf.start_event: self.perf.record_inference_time_gpu(self.perf.start_event,self.perf.end_event)
                            
                            input_tensor_for_postproc_shape_ref = batch_argb_for_viz_and_model if batch_argb_for_viz_and_model is not None else yolo_model_input

                            print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting _process_inference_results.", flush=True)
                            # CORRECTED KEYWORD ARGUMENT HERE
                            batch_log_entries, batch_viz_output_for_display = self._process_inference_results(
                                results_yolo=yolo_results, 
                                input_batch_tensor_for_shape_ref=input_tensor_for_postproc_shape_ref, 
                                frame_numbers=frame_numbers_batch_abs, 
                                frame_times=frame_times_batch_abs, 
                                start_datetime_chunk=start_datetime_chunk
                            )
                            print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished _process_inference_results. Extending logs.", flush=True)
                            all_detections_structured_log.extend(batch_log_entries)
                            if self.perf: print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Recording detection count.", flush=True); self.perf.record_detection(sum(len(r.boxes) for r in yolo_results if hasattr(r,'boxes') and r.boxes))
                            
                            if ENABLE_VISUALIZATION:
                                if _viz_enabled_this_run and self.writer_instance is not None and batch_argb_for_viz_and_model is not None:
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting GPU overlays.", flush=True)
                                    if self.perf: self.perf.start_timer('drawing_gpu')
                                    visualized_batch_argb_float=add_gpu_overlays(batch_argb_for_viz_and_model, batch_viz_output_for_display, self.zone_tracker.zones)
                                    if self.perf: self.perf.end_timer('drawing_gpu')
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished GPU overlays.", flush=True)
                                    if visualized_batch_argb_float is not None:
                                        print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting writer.put loop (GPU).", flush=True)
                                        for frame_idx_in_batch, frame_tensor_argb_float in enumerate(visualized_batch_argb_float):
                                            frame_to_encode_argb_uint8=(frame_tensor_argb_float.clamp(0,1)*255.0).byte()
                                            self.writer_instance.put(frame_to_encode_argb_uint8)
                                        print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished writer.put loop (GPU).", flush=True)
                                elif _cpu_viz_enabled_this_run and self.video_write_queue is not None: 
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting CPU frame write to queue.", flush=True)
                                    if self.perf: self.perf.start_timer('video_write_cpu');
                                    for pf_cpu in batch_viz_output_for_display: 
                                        if pf_cpu is not None: 
                                            try: self.video_write_queue.put(pf_cpu, block=True, timeout=2.0)
                                            except queue.Full: print(f"WARN: CPU Video writer queue full for batch {log_batch_id_str}, frame dropped.", flush=True)
                                    if self.perf: self.perf.end_timer('video_write_cpu')
                                    print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished CPU frame write to queue.", flush=True)

                        else: print(f"Warning: Preprocessing or model input prep failed for batch {log_batch_id_str}. Skipping inference.", flush=True)
                        
                        print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Starting batch perf recording.", flush=True)
                        batch_time=time.monotonic()-batch_start_mono; frames_processed_count+=num_processed_in_batch
                        if self.perf: self.perf.record_batch_processed(num_processed_in_batch,batch_time); print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished record_batch_processed.", flush=True)
                        
                        current_mono_time=time.monotonic()
                        if current_mono_time-self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - ENTERING PERIODIC CLEANUP. Current mono time: {current_mono_time}, Last cleanup: {self.last_cleanup_time}", flush=True)
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_f_time = frame_times_batch_abs[-1] if frame_times_batch_abs else 0
                            cleanup_tracking_data(self.tracker, self.zone_tracker, last_f_time)
                            print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - Finished cleanup_tracking_data.", flush=True)
                            cleanup_memory()
                            self.last_cleanup_time = current_mono_time
                            if self.perf: self.perf.end_timer('memory_cleanup')
                            print(f"DEBUG_VP: {log_batch_id_str} ({base_video_name}) - EXITED PERIODIC CLEANUP.", flush=True)
                        
                        if progress_callback and self.perf:
                            p_stats=self.perf.get_progress(); base_prog=(current_chunk_idx_disp-1)/total_chunks_disp; chunk_prog=p_stats.get('percent',0)/100.0; overall=min(1.0,base_prog+(chunk_prog/total_chunks_disp))
                            status=f"Chunk {current_chunk_idx_disp}/{total_chunks_disp}: {p_stats['percent']:.1f}% (FPS:{p_stats['fps']:.1f}|ETA:{p_stats['eta']})"
                            try: progress_callback(overall, status)
                            except Exception as cb_err: print(f"Warning: Gradio progress_callback error: {cb_err}", flush=True)
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0):
                            prog_stats_console=self.perf.get_progress(); active_trk_count=self.tracker.get_active_vehicle_count() if hasattr(self.tracker,'get_active_vehicle_count') else 'N/A'
                            print(f"\rChunk {current_chunk_idx_disp}/{total_chunks_disp} Prog:{prog_stats_console['percent']:.1f}%|FPS:{prog_stats_console['fps']:.1f}|Active:{active_trk_count}|ETA:{prog_stats_console['eta']} ", end="", flush=True)
                            last_progress_print_time = current_mono_time
                        
                        current_batch_first_frame_num_logging = frame_numbers_batch_abs[0] if frame_numbers_batch_abs else "N/A_BeforeClear"
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before .clear() on lists.", flush=True)
                        frames_batch_input_list.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                        self.frames_batch_input_list_for_cpu_draw.clear() 
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - After .clear() on lists.", flush=True)
                        
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before deleting 'batch_argb_for_viz_and_model'.", flush=True)
                        if 'batch_argb_for_viz_and_model' in locals() and batch_argb_for_viz_and_model is not None: del batch_argb_for_viz_and_model
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before deleting 'yolo_model_input'.", flush=True)
                        if 'yolo_model_input' in locals() and yolo_model_input is not None: del yolo_model_input
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before deleting 'yolo_results'.", flush=True)
                        if 'yolo_results' in locals() and yolo_results is not None: del yolo_results
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before deleting 'batch_log_entries'.", flush=True)
                        if 'batch_log_entries' in locals(): del batch_log_entries
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before deleting 'batch_viz_output_for_display'.", flush=True)
                        if 'batch_viz_output_for_display' in locals(): del batch_viz_output_for_display
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - All local batch vars potentially deleted.", flush=True)
                        
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - Before cleanup_memory().", flush=True)
                        cleanup_memory()
                        print(f"DEBUG_VP: Batch with first frame {current_batch_first_frame_num_logging} ({base_video_name}) - After cleanup_memory().", flush=True)

                    if self.frame_read_queue:
                        print(f"DEBUG_VP: Main loop iter for frame {frame_num_abs} ({base_video_name}) - Before frame_read_queue.task_done().", flush=True)
                        self.frame_read_queue.task_done()
                        print(f"DEBUG_VP: Main loop iter for frame {frame_num_abs} ({base_video_name}) - After frame_read_queue.task_done().", flush=True)
                except queue.Empty:
                    if self.reader_thread and not self.reader_thread.is_alive() and self.frame_read_queue.empty():
                        print(f"\n_process_single_video_file: Reader thread finished and queue empty. Exiting loop.", flush=True); break
                    else: continue
                except Exception as e_loop:
                    print(f"\n_process_single_video_file: ERROR during processing loop: {e_loop}", flush=True); import traceback; traceback.print_exc();
                    if self.stop_event: self.stop_event.set(); break
            
            if frames_batch_input_list: 
                final_batch_id_for_log = frame_numbers_batch_abs[0] if frame_numbers_batch_abs else "FinalBatch_Unknown"
                print(f"\n_process_single_video_file: Processing final batch of {len(frames_batch_input_list)} frames starting with {final_batch_id_for_log}...", flush=True)
                batch_start_mono=time.monotonic()
                
                if _cpu_viz_enabled_this_run:
                    self.frames_batch_input_list_for_cpu_draw = [f.copy() for f in frames_batch_input_list if isinstance(f, np.ndarray)]

                yolo_model_input_final = None; batch_argb_for_viz_final = None
                if use_nvidia_reader and isinstance(frames_batch_input_list[0], torch.Tensor):
                    if self.perf: self.perf.start_timer('preprocessing'); batch_argb_for_viz_final=self._preprocess_batch(frames_batch_input_list); self.perf.end_timer('preprocessing')
                    if batch_argb_for_viz_final is not None and batch_argb_for_viz_final.nelement()>0: yolo_model_input_final = batch_argb_for_viz_final[:, 1:4, :, :]
                else: 
                    tensors_cpu_final_b = []
                    for cpu_frame_item_fb in frames_batch_input_list:
                        if cpu_frame_item_fb is None or not isinstance(cpu_frame_item_fb, np.ndarray): continue
                        img_cpu_fb=cv2.resize(cpu_frame_item_fb,(self.model_input_size_config,self.model_input_size_config)); img_cpu_fb=cv2.cvtColor(img_cpu_fb,cv2.COLOR_BGR2RGB)
                        img_cpu_fb=img_cpu_fb.transpose(2,0,1); img_cpu_fb=np.ascontiguousarray(img_cpu_fb); tensors_cpu_final_b.append(img_cpu_fb)
                    if tensors_cpu_final_b:
                        yolo_model_input_final =torch.from_numpy(np.stack(tensors_cpu_final_b)).to(self.device)
                        dtype_cpu_fb=torch.float16 if (MIXED_PRECISION and self.device=='cuda') else torch.float32
                        yolo_model_input_final = yolo_model_input_final.to(dtype_cpu_fb)/255.0
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished preprocessing.", flush=True)
                
                num_processed_in_final_batch=len(frames_batch_input_list)
                if yolo_model_input_final is not None and yolo_model_input_final.nelement()>0:
                    print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Starting model inference.", flush=True)
                    results_final=self.model(yolo_model_input_final,imgsz=self.model_input_size_config,verbose=False)
                    print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished model inference.", flush=True)
                    
                    input_tensor_for_postproc_viz_fb = batch_argb_for_viz_final if batch_argb_for_viz_final is not None else yolo_model_input_final
                    print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Starting _process_inference_results.", flush=True)
                    batch_log_final, batch_viz_data_final =self._process_inference_results(
                        results_yolo=results_final,
                        input_batch_tensor_for_shape_ref=input_tensor_for_postproc_viz_fb, # CORRECTED
                        frame_numbers=frame_numbers_batch_abs,
                        frame_times=frame_times_batch_abs,
                        start_datetime_chunk=start_datetime_chunk
                    )
                    print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished _process_inference_results.", flush=True)
                    all_detections_structured_log.extend(batch_log_final)
                    
                    if ENABLE_VISUALIZATION : 
                        if _viz_enabled_this_run and self.writer_instance is not None and batch_argb_for_viz_final is not None: 
                            print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Starting GPU overlays.", flush=True)
                            if self.perf: self.perf.start_timer('drawing_gpu')
                            visualized_batch_argb_float_final=add_gpu_overlays(batch_argb_for_viz_final, batch_viz_data_final, self.zone_tracker.zones)
                            if self.perf: self.perf.end_timer('drawing_gpu')
                            print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished GPU overlays.", flush=True)
                            if visualized_batch_argb_float_final is not None:
                                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Starting writer.put loop (GPU).", flush=True)
                                for frame_tensor_argb_float_f in visualized_batch_argb_float_final:
                                    frame_to_encode_argb_uint8_f=(frame_tensor_argb_float_f.clamp(0,1)*255.0).byte()
                                    self.writer_instance.put(frame_to_encode_argb_uint8_f)
                                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished writer.put loop (GPU).", flush=True)
                        elif _cpu_viz_enabled_this_run and self.video_write_queue is not None: 
                            print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Starting CPU frame write to queue.", flush=True)
                            if self.perf: self.perf.start_timer('video_write_cpu');
                            for pf_cpu_f in batch_viz_data_final: 
                                if pf_cpu_f is not None: 
                                    try: self.video_write_queue.put(pf_cpu_f, block=True, timeout=2.0)
                                    except queue.Full: print(f"WARN: CPU Video writer queue full for final batch {log_batch_id_str}, frame dropped.", flush=True)
                            if self.perf: self.perf.end_timer('video_write_cpu')
                            print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Finished CPU frame write to queue.", flush=True)
                
                frames_processed_count+=num_processed_in_final_batch
                if self.perf: self.perf.record_batch_processed(num_processed_in_final_batch, time.monotonic() - batch_start_mono)

                print("\n--- Processed Frame Index Summary (Final Batch) ---", flush=True)
                if processed_frame_indices_log: 
                    processed_frame_indices_log.sort(); count_f = len(processed_frame_indices_log); min_f = processed_frame_indices_log[0]; max_f = processed_frame_indices_log[-1]
                    print(f"Total frames processed (logged overall): {count_f}", flush=True); print(f"Min frame index: {min_f}", flush=True); print(f"Max frame index: {max_f}", flush=True)
                else: print("No frames were processed (logged for final batch).", flush=True)
                print("--- End Processed Frame Index Summary (Final Batch) ---", flush=True)
                
                num_tasks_to_done_final_b=len(frames_batch_input_list)
                if self.frame_read_queue:
                    for _ in range(num_tasks_to_done_final_b):
                        try: self.frame_read_queue.task_done()
                        except ValueError: break
                
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Before clearing lists (final).", flush=True)
                frames_batch_input_list.clear(); frame_numbers_batch_abs.clear(); frame_times_batch_abs.clear()
                self.frames_batch_input_list_for_cpu_draw.clear()
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - After clearing lists (final).", flush=True)
                
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Before deleting tensors (final).", flush=True)
                if 'batch_argb_for_viz_final' in locals() and batch_argb_for_viz_final is not None: del batch_argb_for_viz_final
                if 'yolo_model_input_final' in locals() and yolo_model_input_final is not None: del yolo_model_input_final
                if 'results_final' in locals() and results_final is not None: del results_final
                if 'batch_log_final' in locals(): del batch_log_final
                if 'batch_viz_data_final' in locals(): del batch_viz_data_final
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - All local batch vars potentially deleted (final).", flush=True)
                
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - Before cleanup_memory() (final).", flush=True)
                cleanup_memory()
                print(f"DEBUG_VP (Final): {final_batch_id_for_log} ({base_video_name}) - After cleanup_memory() (final).", flush=True)
                print(f"Finished processing final batch. Total frames processed by inference: {frames_processed_count}", flush=True)
            
            print(f"_process_single_video_file: Finalizing for {base_video_name}...", flush=True)
            if self.tracker and hasattr(self.tracker, 'active_vehicles') and self.tracker.active_vehicles:
                 print(f"Forcing exit for {len(self.tracker.active_vehicles)} remaining active vehicles...", flush=True)
                 last_ts_in_chunk = start_datetime_chunk
                 if self.last_processed_frame_time_abs_in_loop is not None:
                     last_ts_in_chunk = start_datetime_chunk + timedelta(seconds=self.last_processed_frame_time_abs_in_loop)
                 
                 last_fnum_for_force_exit = (processed_frame_indices_log[-1] if processed_frame_indices_log else (total_frames_file -1 if total_frames_file > 0 else 0))
                 for v_id_force in list(self.tracker.active_vehicles.keys()):
                     if self.tracker.force_exit_vehicle(v_id_force, last_ts_in_chunk):
                         if self.perf: self.perf.record_vehicle_exit('forced_exit')
                         p_data=next((p for p in reversed(getattr(self.tracker,'completed_paths',[])) if p['id']==v_id_force and p['status']=='forced_exit'),None)
                         if p_data: all_detections_structured_log.append({'timestamp_dt':last_ts_in_chunk,'vehicle_id':v_id_force,'vehicle_type':p_data.get('type','Unknown'),'event_type':'FORCED_EXIT','direction_from':p_data.get('entry_direction','UNKNOWN'),'direction_to':'FORCED','status':'forced_exit','time_in_intersection':p_data.get('time_in_intersection','N/A'),'frame_number':last_fnum_for_force_exit})
            
            print(f"_process_single_video_file: Signaling stop_event for {base_video_name}...", flush=True);
            if self.stop_event: self.stop_event.set()
            
            if self.writer_instance is not None : 
                print(f"Finalize: Sending EOS (None) to Nvidia writer queue for {base_video_name}...", flush=True); 
                self.writer_instance.put(None) 
            elif self.video_write_queue and ENABLE_VISUALIZATION: 
                print(f"Finalize: Sending EOS (None) to CPU writer queue for {base_video_name}...", flush=True); 
                try: 
                    self.video_write_queue.put(None, timeout=5.0)
                except queue.Full: 
                    print("WARN: Could not signal CPU writer queue (full).", flush=True)

            if self.reader_thread and self.reader_thread.is_alive():
                print(f"Finalize: Joining reader thread for {base_video_name}...", flush=True); self.reader_thread.join(timeout=10)
                if self.reader_thread.is_alive(): print(f"Warning: Reader thread join timed out for {base_video_name}.", flush=True)
                else: print(f"Reader thread joined for {base_video_name}.", flush=True)
            
            mux_success_this_file = True 
            if self.writer_instance is not None: # Nvidia Writer
                 print(f"Finalize: Stopping Nvidia writer (includes muxing) for {base_video_name}...", flush=True); mux_success_this_file = self.writer_instance.stop() 
                 if mux_success_this_file and hasattr(self.writer_instance, 'output_path'): self.final_output_path_current_file = self.writer_instance.output_path
                 else: self.final_output_path_current_file = None; print(f"Warning: Muxing failed or Nvidia writer stopped improperly for {base_video_name}.", flush=True)
            elif self.writer_thread and self.writer_thread.is_alive(): # CPU Writer
                print(f"Finalize: Joining CPU writer thread for {base_video_name}...", flush=True); self.writer_thread.join(timeout=30)
                if self.writer_thread.is_alive(): print(f"Warning: CPU Writer thread did not finish in time.", flush=True)
                else: print(f"CPU Writer thread joined for {base_video_name}.", flush=True)
            elif ENABLE_VISUALIZATION : print(f"Finalize: Writer was not initialized (ENABLE_VISUALIZATION is True but no writer instance/thread).", flush=True)
            
            total_time_taken_file=0.0; fps_file=0.0; completed_paths_count_valid=0
            if self.perf: self.perf.end_processing(); self.perf.print_summary(); total_time_taken_file=self.perf.total_time; fps_file=frames_processed_count/total_time_taken_file if total_time_taken_file>0 else 0
            final_completed_paths_for_report = []
            if hasattr(self, 'tracker') and self.tracker:
                 if hasattr(self.tracker, 'get_completed_paths'): final_completed_paths_for_report = self.tracker.get_completed_paths()
                 elif hasattr(self.tracker, 'completed_paths'): final_completed_paths_for_report = self.tracker.completed_paths
            for path_rep in final_completed_paths_for_report:
                status=path_rep.get('status'); entry_dir=path_rep.get('entry_direction'); exit_dir=path_rep.get('exit_direction')
                if status=='exited' and entry_dir and entry_dir not in ['UNKNOWN',None] and exit_dir and exit_dir not in ['UNKNOWN','TIMEOUT','FORCED',None]: completed_paths_count_valid+=1
            print(f"Generating report for {base_video_name}...", flush=True)
            excel_file_path=create_excel_report(completed_paths_data=final_completed_paths_for_report,start_datetime=start_datetime_chunk,primary_direction=primary_direction,video_path=video_path,output_path_override=output_path_override)
            result_msg_this_file = f"✅ Processing completed for {base_video_name}.\n"
            if ENABLE_VISUALIZATION:
                if self.final_output_path_current_file and "ERROR" not in self.final_output_path_current_file: result_msg_this_file+=f"Output video segment: '{self.final_output_path_current_file}'.\n"
                elif self.final_output_path_current_file and "ERROR" in self.final_output_path_current_file: result_msg_this_file+=f"Output video segment generation failed: {self.final_output_path_current_file}.\n"
                else: result_msg_this_file+=f"Video output was enabled but no final path determined or writer failed.\n"
            if excel_file_path: result_msg_this_file+=f"Report segment: {excel_file_path}\n"
            else: result_msg_this_file+=f"Report segment generation failed for {base_video_name}.\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_msg_this_file+=f"Perf charts: 'performance_charts/'\n"
            result_msg_this_file+=f"--- Summary Stats for {base_video_name} ---\n"; result_msg_this_file+=f"STAT_FRAMES_PROCESSED={frames_processed_count}\n"; result_msg_this_file+=f"STAT_TIME_SECONDS={total_time_taken_file:.2f}\n"; result_msg_this_file+=f"STAT_FPS={fps_file:.2f}\n"; result_msg_this_file+=f"STAT_COMPLETED_PATHS={completed_paths_count_valid}\n"; result_msg_this_file+=f"--- End Stats ---"
            print(f"_process_single_video_file: Finished successfully for {base_video_name}.", flush=True)
            return result_msg_this_file
        except Exception as e_outer:
            print(f"\nFATAL ERROR during _process_single_video_file for {base_video_name}: {e_outer}", flush=True); import traceback; traceback.print_exc();
            if hasattr(self,'stop_event') and self.stop_event and not self.stop_event.is_set(): print("Signalling stop event due to outer exception...", flush=True); self.stop_event.set()
            if hasattr(self,'writer_instance') and self.writer_instance:
                try: print("Attempting to stop Nvidia writer after outer exception...", flush=True); self.writer_instance.stop(cleanup_raw_file=True)
                except Exception as e_stop: print(f"Exception while trying to stop Nvidia writer during error handling: {e_stop}", flush=True)
            if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive():
                 print("Attempting to stop/join CPU writer after outer exception...", flush=True)
                 if self.video_write_queue:
                     try: self.video_write_queue.put(None, timeout=1.0)
                     except: pass
                 self.writer_thread.join(timeout=5)
            return f"❌ Error processing {base_video_name}: {e_outer}"
        finally:
            print(f"_process_single_video_file: Entering finally block for {base_video_name}...", flush=True)
            if hasattr(self,'reader_thread') and self.reader_thread and self.reader_thread.is_alive():
                print(f"Finally: Joining reader thread for {base_video_name}...", flush=True); self.reader_thread.join(timeout=5)
                if self.reader_thread.is_alive(): print(f"Warning: Reader thread join timed out in finally block for {base_video_name}.", flush=True)
            if hasattr(self,'writer_instance') and self.writer_instance: 
                print(f"Finally: Nvidia Writer instance existed for {base_video_name}. Stop should have been called.", flush=True)
            if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive():
                 print(f"Finally: Joining CPU writer thread for {base_video_name}...", flush=True)
                 self.writer_thread.join(timeout=5)
                 if self.writer_thread.is_alive(): print(f"Warning: CPU writer thread join timed out in finally.", flush=True)

            self.tracker=None; self.zone_tracker=None; self.perf=None; self.frame_read_queue=None; self.reader_thread=None
            self.writer_instance=None; self.reader_instance=None; self.stop_event=None; self.reader_dimensions_ready_event=None
            self.final_output_path_current_file=None; self.detection_zones_polygons=None; self.video_write_queue = None
            self.last_processed_frame_time_abs_in_loop = None 
            self._batch_processed_count_for_debug = 0 
            cleanup_memory()
            print(f"Finished resource cleanup for {base_video_name}", flush=True)
            if self.processing_lock.locked():
                 try: self.processing_lock.release(); print("Processing lock released in finally.", flush=True)
                 except RuntimeError as release_err: print(f"Warn: Error releasing lock in finally: {release_err}", flush=True)

# --- End of VideoProcessor class ---
