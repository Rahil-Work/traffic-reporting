# project/core/utils.py
import os
import torch
import gc
from datetime import datetime
from config import DEBUG_MODE, VALID_MOVEMENTS

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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

def is_valid_movement(from_dir, to_dir):
    """Check if movement from from_dir to to_dir is valid based on config."""
    from_dir_norm = normalize_direction(from_dir)
    to_dir_norm = normalize_direction(to_dir)

    if not from_dir_norm or not to_dir_norm: return False
    if from_dir_norm == to_dir_norm: return False # U-turn
    if from_dir_norm not in VALID_MOVEMENTS: return False

    valid = to_dir_norm in VALID_MOVEMENTS.get(from_dir_norm, [])
    if not valid and DEBUG_MODE:
         debug_print(f"Invalid movement: '{to_dir_norm}' is not a valid exit from '{from_dir_norm}'. Valid: {VALID_MOVEMENTS.get(from_dir_norm)}")
    return valid

def setup_thread_affinity(perf_cores=8, eff_cores=4):
    """Attempt to set thread affinity for hybrid CPUs (best effort)."""
    if not PSUTIL_AVAILABLE:
        debug_print("psutil not found, cannot set thread affinity.")
        return
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    debug_print("Performed memory cleanup (GC + CUDA Cache)")