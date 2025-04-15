# project/main.py
import os
import sys

# Import specific needed values from config
from config import (
    LINE_MODE, PROCESSING_MODE, GRADIO_SERVER_PORT, GRADIO_SHARE,
    INPUT_VIDEO_PATH, START_DATE, START_TIME
)
from core.utils import setup_torch_global_settings, setup_thread_affinity

# Conditionally import based on LINE_MODE
if LINE_MODE == 'interactive':
    from interface.gradio_app import create_interface
else:
    from core.video_processor import VideoProcessor
    create_interface = None

# --- Main Execution Logic ---
def run_app():
    print("--- Starting Vehicle Detection Application ---")
    print(f"Processing Mode: {PROCESSING_MODE.upper()}")
    print(f"Line Mode: {LINE_MODE.upper()}")

    setup_torch_global_settings()
    setup_thread_affinity()

    if LINE_MODE == 'interactive':
        # --- Run Gradio Interface ---
        # (Gradio logic remains the same)
        if create_interface is None:
             print("ERROR: Gradio components not loaded.", file=sys.stderr); sys.exit(1)
        try:
            print("\nStarting Gradio interface for interactive setup...")
            interface = create_interface()
            interface.launch(server_name="127.0.0.1", server_port=GRADIO_SERVER_PORT, share=GRADIO_SHARE)
            print("Gradio server stopped.")
        except ImportError as e: print(f"Error: Missing Gradio: {e}", file=sys.stderr); sys.exit(1)
        except Exception as e: print(f"Error launching Gradio: {e}", file=sys.stderr); import traceback; traceback.print_exc(); sys.exit(1)

    elif LINE_MODE == 'hardcoded':
        print("\nStarting hardcoded processing...")
        # --- Validate required config values ---
        if not os.path.exists(INPUT_VIDEO_PATH):
             print(f"ERROR: Input video file not found at: {INPUT_VIDEO_PATH}", file=sys.stderr)
             sys.exit(1)

        print(f"Using Input Video: {INPUT_VIDEO_PATH}")
        print(f"Using Start Date:  {START_DATE}")
        print(f"Using Start Time:  {START_TIME}")

        try:
            processor = VideoProcessor()
            result_message = processor.process_video(
                video_path=INPUT_VIDEO_PATH,
                start_date_str=START_DATE,
                start_time_str=START_TIME
            )
            print("\n--- Processing Result ---")
            print(result_message)
            print("-------------------------")
        except Exception as e:
            print(f"\n--- ERROR during hardcoded processing ---", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            print("-----------------------------------------", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Invalid LINE_MODE '{LINE_MODE}'. Choose 'interactive' or 'hardcoded'.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_app()