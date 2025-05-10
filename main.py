# project/main.py
import os
import sys
import threading

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
            processing_completed_successfully = False

            for status_update in processor.process_video(
                video_path=INPUT_VIDEO_PATH,
                start_date_str=START_DATE,
                start_time_str=START_TIME,
                primary_direction=DEFAULT_PRIMARY_DIRECTION.lower() # Pass lowercase
            ):
                if isinstance(status_update, dict) and 'final_message' in status_update:
                    final_result_message = status_update['final_message']
                    print("\n--- Processing Result (from main.py) ---", flush=True)
                    print(final_result_message, flush=True)
                    print("-------------------------", flush=True)
                    if "✅" in final_result_message: # Simple check for success
                        processing_completed_successfully = True
                elif isinstance(status_update, str):
                    print(f"STATUS (from main.py): {status_update}", flush=True)
            print("Main: processor.process_video() generator exhausted.", flush=True)
            # print("\n--- Processing Result ---")
            # print(final_result_message) # Print the final comprehensive message
            print("-------------------------")
        except Exception as e:
            print(f"\n--- ERROR during hardcoded processing ---", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            print("-----------------------------------------", file=sys.stderr)
            final_result_message = f"Error during processing: {e}"
            # sys.exit(1)
        finally:
            print("Main: Entering main's finally block.", flush=True)
            # Lock release in VideoProcessor's finally should handle it.
            # Releasing here again might cause issues if already released.
            # if processor and hasattr(processor, 'processing_lock') and processor.processing_lock.locked():
            #     try:
            #         processor.processing_lock.release()
            #         print("Main: Processing lock released in main's finally block (if held).", flush=True)
            #     except RuntimeError: 
            #         print("Main: Attempted to release lock in main's finally, but was not held.", flush=True)
            #         pass 
            
            print(f"Main: Video processing loop finished. Success: {processing_completed_successfully}", flush=True)
            print(f"Main: Final result message before exit: {final_result_message}", flush=True)
    else:
        print(f"Error: Invalid LINE_MODE '{LINE_MODE}'. Choose 'interactive' or 'hardcoded'.", file=sys.stderr)
        return
        # sys.exit(1)
    print("Main: run_app() is about to finish.", flush=True)

if __name__ == "__main__":
    print("Main: Script execution started.", flush=True)
    try:
        run_app()
    except Exception as e_main:
        print(f"FATAL ERROR in main execution: {e_main}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("Main: Script __main__ finally block reached.", flush=True)
        
        # ---- DEBUG ACTIVE THREADS ----
        print("Main: Listing active threads before sys.exit()...", flush=True)
        active_threads = threading.enumerate()
        if not active_threads or len(active_threads) <= 1 : # MainThread is usually 1
             print("  No unexpected active threads found (or only MainThread).", flush=True)
        for i, th in enumerate(active_threads):
            print(f"  Thread {i+1}: Name='{th.name}', Daemon={th.isDaemon()}, Alive={th.is_alive()}", flush=True)
        # ---- END DEBUG ----
        
        print("Main: Attempting sys.exit(0)...", flush=True)
        sys.exit(0) 