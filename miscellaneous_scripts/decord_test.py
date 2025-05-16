import decord
import torch # Ensure PyTorch is imported to set up CUDA context properly for decord

# Optional: Tell decord to use PyTorch tensors for its GPU arrays
decord.bridge.set_bridge('torch')

try:
    # Attempt to initialize VideoReader with GPU context
    # Replace 'your_video.mp4' with an actual video file path
    # gpu_id is 0 for the first GPU
    vr = decord.VideoReader("C:/Users/EMAAN/Documents/YOLO/5 minute test - 4 Way Intersection.mp4", ctx=decord.gpu(0))
    print("Decord: Successfully opened video with GPU context.")

    # Try to get a frame
    if len(vr) > 0:
        frame_gpu = vr[0] # This should be a torch.Tensor on GPU
        print(f"Decord: Frame 0 retrieved as a {type(frame_gpu)} on device {frame_gpu.device}")
        print(f"Decord: Frame shape: {frame_gpu.shape}, dtype: {frame_gpu.dtype}")
        # Typical output is HWC, uint8, e.g., (1080, 1920, 3) on GPU
    else:
        print("Decord: Video is empty.")

except RuntimeError as e:
    print(f"Decord: Failed to open video with GPU context: {e}")
    print("Decord: Falling back or using CPU version.")
    # vr = decord.VideoReader('your_video.mp4', ctx=decord.cpu())
    # ...
except Exception as e:
    print(f"An unexpected error occurred: {e}")