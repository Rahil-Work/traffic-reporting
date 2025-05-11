from video_reader import PyVideoReader
from time import perf_counter
import numpy as np
import cv2
import math

# --- User-defined Benchmark Parameters ---
filename = "C:/Users/EMAAN/Documents/YOLO/12 hour test - T junction.mp4"

MODEL_INPUT_SIZE = 416
RESIZE_SHORTER_SIDE = MODEL_INPUT_SIZE
RESIZE_LONGER_SIDE = MODEL_INPUT_SIZE

# --- FPS and Selection Logic ---
ORIGINAL_FPS_FROM_VIDEO = 25
TARGET_FPS_FOR_SELECTION = 20 # Frames per second we want to process
DECODE_SEGMENT_SIZE = 7500 # How many frames to decode in one vr.decode(start, end) call

# --- Get actual video properties ---
actual_total_frames = 0
try:
    # Use a context manager if PyVideoReader supports it (good practice)
    with PyVideoReader(filename) as temp_vr_for_props:
        info = temp_vr_for_props.get_info()
        actual_total_frames = int(info.get('frame_count', 0))
        fps_str = info.get('fps', '0.0')
        if fps_str:
            try: ORIGINAL_FPS_FROM_VIDEO = float(fps_str)
            except ValueError: pass
except Exception as e:
    print(f"Could not get props from PyVideoReader ({e}), trying OpenCV...")

if actual_total_frames <= 0 or ORIGINAL_FPS_FROM_VIDEO <= 0:
    cap_check = cv2.VideoCapture(filename)
    if cap_check.isOpened():
        if actual_total_frames <= 0:
            actual_total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        if ORIGINAL_FPS_FROM_VIDEO <= 0:
            ORIGINAL_FPS_FROM_VIDEO = cap_check.get(cv2.CAP_PROP_FPS)
        cap_check.release()
    else:
        print(f"ERROR: Could not open video {filename} with OpenCV to get properties.")
        exit()

if actual_total_frames <= 0:
    print("ERROR: Could not determine total frames for the video.")
    exit()
if ORIGINAL_FPS_FROM_VIDEO <= 0:
    print("Warning: Could not determine original FPS, assuming 25 for this benchmark.")
    ORIGINAL_FPS_FROM_VIDEO = 25.0

FRAME_INTERVAL = max(1, round(ORIGINAL_FPS_FROM_VIDEO / TARGET_FPS_FOR_SELECTION))

# Generate all selected_raw_indices for the entire video first for comparison
all_selected_raw_indices_for_video = []
for i in range(actual_total_frames):
    if i % FRAME_INTERVAL == 0:
        all_selected_raw_indices_for_video.append(i)

if not all_selected_raw_indices_for_video:
    print("Error: No frames selected. Check video properties and FPS targets.")
    exit()

print(f"--- PyVideoReader Efficiency Benchmark (Segmented Decode) ---")
print(f"Video: {filename}")
print(f"Actual Total Frames: {actual_total_frames}, Actual Original FPS: {ORIGINAL_FPS_FROM_VIDEO:.2f}")
print(f"PyVideoReader resize: shorter_side={RESIZE_SHORTER_SIDE}, longer_side={RESIZE_LONGER_SIDE}")
print(f"Target FPS for selection: {TARGET_FPS_FOR_SELECTION}, Frame interval: {FRAME_INTERVAL}")
print(f"Total *selected* frames to fetch if processed: {len(all_selected_raw_indices_for_video)}")
print(f"Decode segment size for Method 1B: {DECODE_SEGMENT_SIZE} frames")


# --- Method 1B: Segmented vr.decode(start, end) then select from segment ---
print(f"\n--- Benchmarking: Method 1B - Segmented vr.decode(start, end) ---")
# We won't store all frames in memory, just time the processing and count
total_time_m1b = -1
processed_frames_count_m1b = 0
first_frame_m1b = None
vr_m1b = None

try:
    vr_m1b = PyVideoReader(filename, device='cuda', 
                           resize_shorter_side=RESIZE_SHORTER_SIDE, 
                           resize_longer_side=RESIZE_LONGER_SIDE)
    
    overall_start_m1b = perf_counter()
    
    num_segments = math.ceil(actual_total_frames / DECODE_SEGMENT_SIZE)
    current_overall_raw_frame_idx = 0 # Tracks the true raw frame index in the video

    for i in range(num_segments):
        segment_start_raw_idx = i * DECODE_SEGMENT_SIZE
        segment_end_raw_idx = min((i + 1) * DECODE_SEGMENT_SIZE, actual_total_frames) # exclusive end for decode

        if segment_start_raw_idx >= actual_total_frames:
            break

        print(f"  M1B: Decoding segment {i+1}/{num_segments} (frames {segment_start_raw_idx} to {segment_end_raw_idx-1})...")
        segment_decode_start_time = perf_counter()
        # PyVideoReader's end_frame might be inclusive or exclusive. Assuming exclusive based on common Python slicing.
        # If it's inclusive, it should be segment_end_raw_idx - 1
        segment_frames_np = vr_m1b.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx)
        segment_decode_duration = perf_counter() - segment_decode_start_time
        print(f"    Segment decode took: {segment_decode_duration:.3f}s")

        if segment_frames_np is not None and segment_frames_np.shape[0] > 0:
            # Now, select frames from this decoded segment based on FRAME_INTERVAL
            # The indices within segment_frames_np are relative to the start of the segment (0-based)
            for j in range(segment_frames_np.shape[0]):
                # current_overall_raw_frame_idx corresponds to segment_start_raw_idx + j
                current_overall_raw_frame_idx = segment_start_raw_idx + j
                
                if current_overall_raw_frame_idx % FRAME_INTERVAL == 0:
                    # This is a frame we want to "process"
                    processed_frames_count_m1b += 1
                    if first_frame_m1b is None: # Save the very first selected frame
                        first_frame_m1b = segment_frames_np[j].copy()
            
            # Simulate some minimal processing for each selected frame if needed, 
            # but the main cost is vr.decode and then iterating the numpy array.
            # For this benchmark, just counting is enough.
        else:
            print(f"  M1B: Segment {i+1} returned no frames or was empty. Stopping.")
            break
            
    total_time_m1b = perf_counter() - overall_start_m1b

    if processed_frames_count_m1b > 0:
        print(f"\nM1B Summary: Total time for {processed_frames_count_m1b} selected frames (from {num_segments} segments): {total_time_m1b:.4f} sec")
        avg_time_m1b = (total_time_m1b / processed_frames_count_m1b) * 1000
        print(f"Average time per *selected* frame (M1B): {avg_time_m1b:.2f} ms")
        if first_frame_m1b is not None:
             cv2.imwrite("frame_M1B_first_selected.jpg", cv2.cvtColor(first_frame_m1b, cv2.COLOR_RGB2BGR))
             print("Saved frame_M1B_first_selected.jpg")
    else:
        print("M1B: No frames successfully processed.")
        total_time_m1b = -1

except Exception as e:
    print(f"Error during Method 1B benchmark: {e}")
    import traceback
    traceback.print_exc()
    total_time_m1b = -1
finally:
    if vr_m1b: del vr_m1b


# # --- Method 1 (Original Full Decode then Select for reference on THIS machine/video) ---
# print(f"\n--- Benchmarking: Method 1 - Full vr.decode() then select from NumPy ---")
# total_time_m1_full = -1
# processed_frames_count_m1_full = 0
# first_frame_m1_full = None
# vr_m1_full = None
# try:
#     vr_m1_full = PyVideoReader(filename, device='cuda', 
#                                resize_shorter_side=RESIZE_SHORTER_SIDE, 
#                                resize_longer_side=RESIZE_LONGER_SIDE)
    
#     overall_start_m1_full = perf_counter()
    
#     print("  M1_Full: Decoding all frames...")
#     decode_call_start = perf_counter()
#     all_frames_video_np = vr_m1_full.decode() # Decode ENTIRE video
#     decode_call_duration = perf_counter() - decode_call_start
#     print(f"    Full .decode() call took: {decode_call_duration:.3f}s")

#     if all_frames_video_np is not None and all_frames_video_np.shape[0] > 0:
#         # Select from the decoded numpy array
#         for i in range(all_frames_video_np.shape[0]):
#             if i % FRAME_INTERVAL == 0:
#                 processed_frames_count_m1_full += 1
#                 if first_frame_m1_full is None:
#                     first_frame_m1_full = all_frames_video_np[i].copy()
#     else:
#         print("  M1_Full: Full .decode() returned no frames.")
        
#     total_time_m1_full = perf_counter() - overall_start_m1_full
    
#     if processed_frames_count_m1_full > 0:
#         print(f"\nM1_Full Summary: Total time for {processed_frames_count_m1_full} selected frames (after full decode): {total_time_m1_full:.4f} sec")
#         avg_time_m1_full = (total_time_m1_full / processed_frames_count_m1_full) * 1000
#         print(f"Average time per *selected* frame (M1_Full): {avg_time_m1_full:.2f} ms")
#         if first_frame_m1_full is not None:
#              cv2.imwrite("frame_M1_Full_first_selected.jpg", cv2.cvtColor(first_frame_m1_full, cv2.COLOR_RGB2BGR))
#              print("Saved frame_M1_Full_first_selected.jpg")
#     else:
#         print("M1_Full: No frames successfully processed.")
#         total_time_m1_full = -1
        
# except Exception as e:
#     print(f"Error during Method 1_Full benchmark: {e}")
#     import traceback
#     traceback.print_exc()
#     total_time_m1_full = -1
# finally:
#     if vr_m1_full: del vr_m1_full


# --- Final Comparison Summary ---
print("\n--- Overall Benchmark Comparison (Full vs Segmented Decode) ---")
valid_m1b = total_time_m1b != -1 and processed_frames_count_m1b > 0
# valid_m1_full = total_time_m1_full != -1 and processed_frames_count_m1_full > 0

if valid_m1b:
    avg_m1b_overall = (total_time_m1b / processed_frames_count_m1b) * 1000
    print(f"Method 1B (Segmented Decode): Total={total_time_m1b:.3f}s, Avg/selected_frame={avg_m1b_overall:.2f}ms, SelectedFrames={processed_frames_count_m1b}")
else:
    print("Method 1B (Segmented Decode): Did not complete successfully or fetched no frames.")

# if valid_m1_full:
#     avg_m1_full_overall = (total_time_m1_full / processed_frames_count_m1_full) * 1000
#     print(f"Method 1_Full (Full Decode):  Total={total_time_m1_full:.3f}s, Avg/selected_frame={avg_m1_full_overall:.2f}ms, SelectedFrames={processed_frames_count_m1_full}")
# else:
#     print("Method 1_Full (Full Decode):  Did not complete successfully or fetched no frames.")

# if valid_m1b and valid_m1_full:
#     if abs(processed_frames_count_m1b - processed_frames_count_m1_full) > 2: # Allow small diff due to segment boundaries
#         print(f"Note: Number of processed frames differs ({processed_frames_count_m1b} vs {processed_frames_count_m1_full}).")

#     if total_time_m1b < total_time_m1_full:
#         print(f"\nSegmented Decode (M1B) was {total_time_m1_full - total_time_m1b:.3f}s faster overall.")
#     elif total_time_m1_full < total_time_m1b:
#         print(f"\nFull Decode (M1_Full) was {total_time_m1b - total_time_m1_full:.3f}s faster overall.")
#     else:
#         print("\nBoth methods took approximately the same overall time.")

#     if first_frame_m1b is not None and first_frame_m1_full is not None:
#         if np.array_equal(first_frame_m1b, first_frame_m1_full):
#             print("First selected frames from both methods are identical.")
#         else:
#             diff_val = np.abs(first_frame_m1b.astype(np.float32) - first_frame_m1_full.astype(np.float32))
#             print(f"First selected frames differ. MAD: {np.mean(diff_val):.2f}")