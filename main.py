# project/main.py
import os
import sys

from config import (
    LINE_MODE, PROCESSING_MODE, GRADIO_SERVER_PORT, GRADIO_SHARE,
    INPUT_VIDEO_PATH, START_DATE, START_TIME, DEFAULT_PRIMARY_DIRECTION
)
from core.utils import setup_torch_global_settings, setup_thread_affinity

if LINE_MODE == 'interactive':
    try:
        from core.video_processor import VideoProcessor
        from interface.gradio_app import create_interface
        GRADIO_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Gradio or core components not found, interactive mode may fail: {e}")
        create_interface = None
        GRADIO_AVAILABLE = False
else: # hardcoded mode
    from core.video_processor import VideoProcessor
    GRADIO_AVAILABLE = False
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
        if not GRADIO_AVAILABLE or create_interface is None:
             print("ERROR: Gradio components not loaded or failed to import.", file=sys.stderr); sys.exit(1)
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
        print(f"Using Primary Direction: {DEFAULT_PRIMARY_DIRECTION}")

        try:
            processor = VideoProcessor()
            if processor is None or processor.model is None:
                print("ERROR: VideoProcessor not initialized or model failed to load.", file=sys.stderr)
                sys.exit(1)
            
            print("\n--- Processing Started ---")
            final_result_message = "Processing did not complete."

            for status_update in processor.process_video(
                video_path=INPUT_VIDEO_PATH,
                start_date_str=START_DATE,
                start_time_str=START_TIME,
                primary_direction=DEFAULT_PRIMARY_DIRECTION.lower() # Pass lowercase
            ):
                if isinstance(status_update, str):
                    # Print intermediate status updates from the processor
                    print(f"STATUS: {status_update}")
                elif isinstance(status_update, dict) and 'final_message' in status_update:
                    # Store the final message when received
                    final_result_message = status_update['final_message']
                    # No break needed here, let generator finish naturally
                else:
                    print(f"Received unexpected status: {status_update}")
            print("\n--- Processing Result ---")
            print(final_result_message) # Print the final comprehensive message
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