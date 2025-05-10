# project/config.py
import os
from datetime import datetime
import torch # Import torch to check GPU capability

PROCESSING_MODE = 'enhanced'
LINE_MODE = 'hardcoded'
ENABLE_VISUALIZATION = False

# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = "C:/Users/EMAAN/Documents/YOLO/5 minute test - 4 Way Intersection.mp4"

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 20
BASE_OUTPUT_DIR = "C:/Users/EMAAN/Documents/YOLO/project/output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
# MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/SGDMed_SGD_LR0p0100_WD0p00050_Cls1p50_Mix0p15_20250418_112036.pt"
MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/SGDMed_SGD_LR0p0100_WD0p00050_Cls1p50_Mix0p15_20250418_112036.engine"
CONF_THRESHOLD = 0.45 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 416 # Keep smaller size for performance

# --- Line & Zone Definitions ---
LINE_POINTS = {
    'north': [(282, 339), 
              (422, 312),
              (371, 240),
              (272, 264)],
    'south': [(290, 446), 
              (605, 335),
              (619, 441),
              (470, 466),
              (361, 476)],
    'east': [(512, 306),  
             (596, 322),
             (628, 214),
             (525, 175)],
    'west': [(252, 378),  
             (244, 442),
             ( 57, 466),
             ( 53, 397)]
}
FRAME_WIDTH = 640; FRAME_HEIGHT = 480
VALID_MOVEMENTS = { 'north': ['south','east','west'], 'south': ['north','east','west'], 'east': ['west','north','south'], 'west': ['east','north','south'] }

# --- Hardware Optimizations ---
# Adjust Batch Size based on chosen MODEL_INPUT_SIZE (320) and GPU VRAM
if MODEL_INPUT_SIZE >= 640: _default_batch = 16 if PROCESSING_MODE == 'standard' else 32
else: _default_batch = 32 if PROCESSING_MODE == 'standard' else 128 # Keep larger for 320
OPTIMAL_BATCH_SIZE = _default_batch

_cpu_count = os.cpu_count()
THREAD_COUNT = 1
# THREAD_COUNT = min(64, _cpu_count * 2) if _cpu_count else 24
MIXED_PRECISION = True if PROCESSING_MODE == 'enhanced' and torch.cuda.is_available() else False
PARALLEL_STREAMS = 16 if PROCESSING_MODE == 'enhanced' else 2

# --- Tracking Configuration --- 
REIDENTIFICATION_TIMEOUT = 10.0
MAX_MATCHING_DISTANCE = 50
TRACKER_CONFIG = ''
TRACKER_TYPE = 'Custom'

# --- Auto Chunking for Long Videos ---
ENABLE_AUTO_CHUNKING = True        # Set to True to enable automatic chunking
AUTO_CHUNK_THRESHOLD_MINUTES = 121   # Threshold duration (in minutes) above which chunking is triggered
AUTO_CHUNK_DURATION_MINUTES = 120    # Desired duration (in minutes) for each chunk
CHUNK_TEMP_DIR = os.path.join(BASE_OUTPUT_DIR, "temp_chunks")
REPORT_TEMP_DIR = os.path.join(BASE_OUTPUT_DIR, "temp_reports")
FFMPEG_PATH = "ffmpeg"

# --- GPU Encoding Settings ---
# Only used if ENABLE_VISUALIZATION is True AND using the GPU pipeline
ENCODER_CODEC = 'h264'       # 'h264' or 'hevc'
ENCODER_BITRATE = '8M'       # e.g., '5M', '8M', '10M' (Mbps)
ENCODER_PRESET = 'p4'        # Quality/speed tradeoff: p1..p7 ('p4' is balanced)
RAW_STREAM_FILENAME = "temp_encoded_stream.raw" # Temp file name for NVENC output
FINAL_VIDEO_EXTENSION = '.mp4' # Or .mkv


# Timeout durations
VEHICLE_TIMEOUTS = {
    'Light Vehicle': 40, 'Motorcycle': 40, 'Minibus Taxi': 50,
    'Short Truck': 50, 'Medium Truck': 55, 'Long Truck': 60,
    'Bus': 55, 'Pedestrian': 65, 'Cyclist': 65,
    'Animal drawn vehicle': 65, 'Person with wheel barrow': 65,
    'default': 40
}
TRACK_HISTORY_LENGTH = 100 # Keep reduced slightly

# Keep configuration for tracker behavior
MAX_CONSECUTIVE_MISSES = 20 # Allow track to persist for X frames without detection

# --- Reporting Configuration ---
VEHICLE_ID_MAP = { 'Light Vehicle': '1', 'Motorcycle': '11', 'Minibus Taxi': '13', 'Short Truck': '2T,2', 'Medium Truck': '2T,3', 'Long Truck': '2T,4', 'Bus': '2B', 'Pedestrian': '9P', 'Cyclist': '9C', 'Animal drawn vehicle': '95', 'Person with wheel barrow': '100' }
ALL_VEHICLE_TYPES = list(VEHICLE_ID_MAP.keys())
REPORTING_DIRECTIONS = { 'From South': ['To North', 'To West', 'To East'], 'From North': ['To South', 'To West', 'To East'], 'From West': ['To North', 'To South', 'To East'], 'From East': ['To North', 'To South', 'To West'] }

# --- Performance & Debugging ---
ENABLE_DETAILED_PERFORMANCE_METRICS = True
DEBUG_MODE = False # Keep False for performance runs
PERFORMANCE_SAMPLE_INTERVAL = 5.0
MEMORY_CLEANUP_INTERVAL = 300

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
print(f"--- Configuration Loaded (Tuned for Exit Accuracy) ---")
print(f"Processing Mode: {PROCESSING_MODE.upper()}")
print(f"Line Definition Mode: {LINE_MODE.upper()}")
if LINE_MODE == 'hardcoded': print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Conf: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
if LINE_MODE == 'hardcoded': print("Using Hardcoded Lines (config.py)")
else: print("Using Interactive Lines (Gradio)")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
# Adjusted print for tracking params
print(f"Re-ID Timeout: {REIDENTIFICATION_TIMEOUT}s, Max Misses: {MAX_CONSECUTIVE_MISSES}") # Removed Max Match Dist
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")