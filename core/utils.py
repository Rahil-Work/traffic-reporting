# project/core/utils.py
import os
import torch
import gc
from datetime import datetime
from config import DEBUG_MODE, VALID_MOVEMENTS
import subprocess # For ffmpeg
import shutil # For directory cleanup
import glob # For finding chunk files
import psutil
import cv2

def debug_print(message):
    """Print debug messages only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"DEBUG: {message}")

def format_timestamp(dt: datetime):
    """Convert datetime to date (YYMMDD) and time (HHMMSSmmm) strings."""
    date_str = dt.strftime('%y%m%d')
    time_str = dt.strftime('%H%M%S')
    ms_str = f"{dt.microsecond // 1000:03d}"
    return date_str, f"{time_str}{ms_str}"

def setup_torch_global_settings():
    """Configure global PyTorch settings for performance."""
    try:
        if torch.cuda.is_available():
            print("Configuring PyTorch for GPU performance...")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()

            if torch.cuda.get_device_capability(0)[0] >= 8: # Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, 'set_float32_matmul_precision'):
                     torch.set_float32_matmul_precision('high')

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            print("PyTorch GPU settings applied.")
        else:
            print("Configuring PyTorch for CPU.")
    except Exception as e:
        print(f"Warning: Could not apply all PyTorch optimizations: {e}")

def get_display_direction(direction):
    """Format direction for display (e.g., 'north' -> 'North')."""
    return direction.capitalize() if direction else "N/A"

def normalize_direction(direction):
    """Convert direction to lowercase for consistent comparison."""
    return direction.lower() if direction else ""

def is_valid_movement(from_dir, to_dir, active_zones):
    """
    Check if movement from from_dir to to_dir is valid based on config
    AND ensures both directions are present in the active_zones list/set.
    Args:
        from_dir (str): The entry direction.
        to_dir (str): The exit direction.
        active_zones (list or set): Collection of lowercase active zone names (e.g., ['north', 'west', 'south']).
    """
    from_dir_norm = normalize_direction(from_dir)
    to_dir_norm = normalize_direction(to_dir)

    # Basic checks
    if not from_dir_norm or not to_dir_norm: return False
    if from_dir_norm == to_dir_norm: return False # U-turn

    # --- Check if both directions are actually active zones ---
    if not active_zones or from_dir_norm not in active_zones or to_dir_norm not in active_zones:
        if DEBUG_MODE:
            debug_print(f"Invalid movement: One or both directions ('{from_dir_norm}' -> '{to_dir_norm}') not in active zones: {active_zones}")
        return False

    # Check against the base VALID_MOVEMENTS rules from config
    if from_dir_norm not in VALID_MOVEMENTS:
        if DEBUG_MODE:
             debug_print(f"Invalid movement: Entry direction '{from_dir_norm}' not found in VALID_MOVEMENTS config.")
        return False

    # Check if the target direction is a valid exit from the entry direction per config rules
    valid_config_targets = VALID_MOVEMENTS.get(from_dir_norm, [])
    is_valid_target = to_dir_norm in valid_config_targets

    if not is_valid_target and DEBUG_MODE:
         debug_print(f"Invalid movement: '{to_dir_norm}' not a valid exit from '{from_dir_norm}' per VALID_MOVEMENTS. Valid targets: {valid_config_targets}")

    return is_valid_target

def setup_thread_affinity(perf_cores=8, eff_cores=4):
    try:
        p = psutil.Process(os.getpid())
        if hasattr(p, 'cpu_affinity'):
            total_logical_cores = psutil.cpu_count(logical=True)
            p_core_indices = [c for c in range(perf_cores * 2) if c < total_logical_cores]
            if p_core_indices:
                p.cpu_affinity(p_core_indices)
                debug_print(f"Set main process affinity to P-cores: {p_core_indices}")
    except Exception as e:
        debug_print(f"Failed to set thread affinity: {e}")

def cleanup_memory():
    """Explicitly run garbage collection and clear CUDA cache."""
    print("DEBUG_UTIL: cleanup_memory - Before gc.collect()", flush=True)
    gc.collect()
    print("DEBUG_UTIL: cleanup_memory - After gc.collect()", flush=True)
    if torch.cuda.is_available():
        print("DEBUG_UTIL: cleanup_memory - Before torch.cuda.synchronize()", flush=True)
        torch.cuda.synchronize() # ADD THIS LINE
        print("DEBUG_UTIL: cleanup_memory - After torch.cuda.synchronize(), Before torch.cuda.empty_cache()", flush=True)
        torch.cuda.empty_cache() 
        print("DEBUG_UTIL: cleanup_memory - After torch.cuda.empty_cache()", flush=True)
    
    # Use your existing debug_print or a direct print for this message
    if DEBUG_MODE:
        # If debug_print doesn't handle flush, use direct print for debugging this part
        print("DEBUG_UTIL: Performed memory cleanup (GC + CUDA Cache)", flush=True)
    else:
        print("Performed memory cleanup (GC + CUDA Cache)", flush=True)

def get_video_properties(video_path):
    """Gets duration (seconds), fps, and frame count using OpenCV."""
    cap = None
    try:
        debug_print(f"Getting properties for video: {video_path}")
        if not os.path.exists(str(video_path)): # Ensure path exists
            print(f"Error: Video file not found at get_video_properties: {video_path}")
            return None, None, None

        cap = cv2.VideoCapture(str(video_path)) # Ensure path is string for OpenCV
        if not cap.isOpened():
            print(f"Error: Cannot open video file with OpenCV: {video_path}")
            return None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
        else: # Handle cases where OpenCV might not get these values
            print(f"Warning: OpenCV could not retrieve valid fps ({fps}) or frame_count ({frame_count}) for {video_path}. Attempting ffprobe.")
            # Fallback to ffprobe if OpenCV fails for duration/fps
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=duration,r_frame_rate,nb_frames',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                output = result.stdout.strip().split('\n')
                if len(output) >= 2: # Expecting duration, r_frame_rate, potentially nb_frames
                    duration_str = output[0]
                    fps_str = output[1]
                    
                    duration = float(duration_str)
                    if '/' in fps_str:
                        num, den = map(float, fps_str.split('/'))
                        fps = num / den if den != 0 else 0.0
                    else:
                        fps = float(fps_str)
                    
                    if len(output) >= 3 and output[2].isdigit(): # nb_frames might not always be available or accurate
                        frame_count = int(output[2])
                    elif duration > 0 and fps > 0: # Estimate frame_count if not provided by ffprobe
                        frame_count = int(duration * fps)
                    else: # If ffprobe also fails significantly
                         print(f"Warning: ffprobe also failed to get sufficient info for {video_path}")
                         return None, None, None

                    debug_print(f"Properties from ffprobe for {video_path}: Duration={duration:.2f}s, FPS={fps:.2f}, Frames={frame_count}")

                else:
                    print(f"Warning: ffprobe output for {video_path} was unexpected: {output}")
                    return None, None, None
            except Exception as e:
                print(f"Error using ffprobe for {video_path}: {e}. OpenCV values used if available, otherwise failing.")
                if not (fps > 0 and frame_count > 0): # If OpenCV also failed
                    return None, None, None
                # If OpenCV got something, use it despite warning above
                duration = frame_count / fps if fps > 0 else 0

        debug_print(f"Video Properties for {video_path}: Duration={duration:.2f}s, FPS={fps:.2f}, Frames={frame_count}")
        return duration, fps, frame_count
    except Exception as e:
        print(f"Error getting video properties for {video_path}: {e}")
        return None, None, None
    finally:
        if cap:
            cap.release()

def split_video_ffmpeg(input_path, output_dir, chunk_duration_sec, ffmpeg_path="ffmpeg", progress_callback=None):
    """Splits video using ffmpeg and returns list of chunk paths."""
    if not os.path.exists(str(input_path)):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if os.path.exists(output_dir):
        print(f"Cleaning up existing temporary chunk directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "chunk_%04d.mp4") # Use %04d for many chunks

    command = [
        ffmpeg_path,
        '-i', str(input_path),
        '-c', 'copy',
        '-map', '0',
        '-segment_time', str(chunk_duration_sec),
        '-f', 'segment',
        '-reset_timestamps', '1',
        '-loglevel', 'error', # Only show errors from ffmpeg
        output_pattern
    ]

    print(f"Running FFmpeg command: {' '.join(command)}")
    if progress_callback:
        # This callback is for before/after, not for live ffmpeg stdout parsing
        progress_callback(0.01, f"Splitting video into ~{chunk_duration_sec}s chunks...")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        # If check=True, it will raise CalledProcessError on non-zero exit.
        # stderr can be checked for warnings even on success if needed.
        if process.stderr:
            debug_print(f"FFmpeg stderr (even on success): {process.stderr}")

        chunk_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*.mp4")))
        if not chunk_files:
            print("Warning: FFmpeg ran but no chunk files were found. Check ffmpeg output/command.")
            # Attempt to read stderr from process object if check=True didn't raise for some reason
            error_output = process.stderr if hasattr(process, 'stderr') else "No stderr captured."
            raise RuntimeError(f"FFmpeg splitting produced no files. FFmpeg output: {error_output}")

        print(f"FFmpeg splitting complete. Generated {len(chunk_files)} chunks.")
        if progress_callback:
             progress_callback(0.05, f"Video split into {len(chunk_files)} chunks.") # Small progress bump
        return chunk_files

    except FileNotFoundError:
         error_msg = f"ERROR: '{ffmpeg_path}' command not found. Ensure FFmpeg is installed and in PATH, or FFMPEG_PATH is set correctly in config.py."
         print(error_msg)
         if progress_callback: progress_callback(0.0, error_msg)
         raise
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR: FFmpeg failed (exit code {e.returncode}).\nFFmpeg stderr: {e.stderr}"
        print(error_msg)
        if progress_callback: progress_callback(0.0, f"FFmpeg splitting failed.")
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"ERROR: An unexpected error occurred during FFmpeg execution: {e}"
        print(error_msg)
        if progress_callback: progress_callback(0.0, "Unexpected error during splitting.")
        raise
