# project/interface/gradio_app.py
import gradio as gr
import os
import re
import time
from config import (
    START_DATE, START_TIME, PROCESSING_MODE, VIDEO_OUTPUT_DIR, LINE_MODE,
    ENABLE_VISUALIZATION # Import for display
)
# Ensure VideoProcessor and GPU_VIZ_ENABLED are correctly determined
try:
    from core.video_processor import VideoProcessor, GPU_VIZ_ENABLED # Import GPU_VIZ_ENABLED
    processor = VideoProcessor()
    PROCESSOR_INITIALIZED = True
except ImportError as e:
    print(f"ERROR: Gradio could not import VideoProcessor or GPU_VIZ_ENABLED: {e}")
    processor = None
    GPU_VIZ_ENABLED = False # Fallback
    PROCESSOR_INITIALIZED = False
except RuntimeError as e: # Catch potential model loading errors during VideoProcessor init
    print(f"FATAL ERROR: VideoProcessor initialization failed: {e}")
    processor = None
    GPU_VIZ_ENABLED = False
    PROCESSOR_INITIALIZED = False

from core.utils import debug_print

# --- Gradio Event Handlers ---
# (Keep your existing polygon drawing handlers: handle_upload, handle_set_direction,
# handle_click, handle_finish_polygon, handle_undo_point, handle_clear_polygons.
# They interact with processor.gradio_polygons etc. and should still largely work.)
def handle_upload(video_path):
    debug_print(f"Gradio: Handling video upload: {video_path}")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized correctly."
    return processor.process_video_upload_for_gradio(video_path)

def handle_set_direction(direction):
    debug_print(f"Gradio: Setting direction for polygon: {direction}")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized."
    frame_with_polys, status = processor.set_gradio_polygon_direction(direction)
    return frame_with_polys, status

def handle_click(evt: gr.SelectData): # Keep type hint if it works for you
    debug_print(f"Gradio: Handling click event at index: {evt.index} for polygon vertex")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized."
    frame_with_polys, status = processor.handle_gradio_polygon_click(evt)
    return frame_with_polys, status

def handle_finish_polygon():
    debug_print(f"Gradio: Finishing current polygon")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized."
    frame_with_polys, status = processor.finalize_current_polygon()
    return frame_with_polys, status

def handle_undo_point():
    debug_print(f"Gradio: Undoing last point")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized."
    frame_with_polys, status = processor.undo_last_polygon_point()
    return frame_with_polys, status

def handle_clear_polygons():
    debug_print(f"Gradio: Clearing all polygons")
    if not PROCESSOR_INITIALIZED or processor is None:
        return None, "Error: Video Processor not initialized."
    frame_with_polys, status = processor.reset_gradio_polygons()
    return frame_with_polys, status


# --- MODIFIED: Process the video ---
def handle_process(video_path, start_date_str, start_time_str, primary_direction_str):
    """
    Handles the video processing request and yields status updates.
    """
    if not PROCESSOR_INITIALIZED or processor is None:
        yield {
            process_status_text: gr.update(value="FATAL: Video Processor not initialized. Cannot process."),
            completion_text: gr.update(value="Video Processor failed to initialize. Check logs."),
            video_output: None, stat_frames: "N/A", stat_time: "N/A", stat_fps: "N/A", stat_completed: "N/A"
        }
        return

    # --- Input Validation ---
    if video_path is None:
        yield { process_status_text: gr.update(value="Error: Please upload a video first."), completion_text: "Error: Please upload a video first."}
        return
    if not primary_direction_str:
        yield { process_status_text: gr.update(value="Error: Please select a Primary Direction."), completion_text: "Error: Please select a Primary Direction."}
        return

    # --- Initialize UI for processing ---
    yield {
        process_status_text: gr.update(value="Initiating processing..."),
        completion_text: "", video_output: None,
        stat_frames: "Processing...", stat_time: "Processing...",
        stat_fps: "Processing...", stat_completed: "Processing..."
    }

    final_result_package = None
    try:
        # VideoProcessor.process_video is now a generator yielding status updates
        # The last yield from process_video should be the full result message.
        for status_or_result in processor.process_video(
            video_path,
            start_date_str,
            start_time_str,
            primary_direction=primary_direction_str.lower(), # Pass lowercased
            progress_callback=None # Simple print/log callback inside processor is enough for now
                                   # Direct Gradio UI updates from callback are complex with blocking calls.
                                   # We rely on processor.process_video yielding status.
        ):
            if isinstance(status_or_result, str): # It's a status update string
                yield {process_status_text: gr.update(value=status_or_result)}
            elif isinstance(status_or_result, dict) and 'final_message' in status_or_result: # Special marker for final result
                final_result_package = status_or_result['final_message']
                debug_print(f"Gradio received final_result_package: {final_result_package[:200]}...") # Log snippet
                break # Exit loop, final processing next
            else: # Fallback for unexpected yield type
                debug_print(f"Gradio: Received unexpected status type: {type(status_or_result)}")
                yield {process_status_text: gr.update(value=str(status_or_result))}


    except Exception as e:
        print(f"Gradio: Error during call to processor.process_video: {e}")
        import traceback; traceback.print_exc()
        final_result_package = f"❌ Unexpected Error during processing: {e}" # Treat as final error message

    # --- Process the final result message ---
    if final_result_package is None:
        final_result_package = "Processing did not yield a final result message."
        yield {process_status_text: gr.update(value=final_result_package)}


    # --- UI Update with final results ---
    frames_processed = "N/A"; time_seconds = "N/A"; fps = "N/A"; completed_paths = "N/A"
    output_video_display_path = None # Path for Gradio Video component
    output_log_display = final_result_package

    if "✅ Processing completed" in final_result_package:
        # Parse stats (same regex as before)
        match_frames = re.search(r"STAT_FRAMES_PROCESSED=(\d+)", final_result_package)
        match_time = re.search(r"STAT_TIME_SECONDS=([\d\.]+)", final_result_package)
        match_fps = re.search(r"STAT_FPS=([\d\.]+)", final_result_package)
        match_paths = re.search(r"STAT_COMPLETED_PATHS=(\d+)", final_result_package)
        if match_frames: frames_processed = match_frames.group(1)
        if match_time: time_seconds = match_time.group(1)
        if match_fps: fps = match_fps.group(1)
        if match_paths: completed_paths = match_paths.group(1)

        # Clean up log for display and find report/video paths
        output_log_display = re.sub(r"--- Summary Stats ---.*--- End Stats ---", "", final_result_package, flags=re.DOTALL).strip()

        # Find report path
        match_report = re.search(r"Consolidated report: (.+\.xlsx)", final_result_package) or \
                       re.search(r"Report: (.+\.xlsx)", final_result_package) or \
                       re.search(r"Report segment: (.+\.xlsx)", final_result_package) # Check for segment path too
        if match_report:
            output_log_display += f"\n\nGenerated Report: {match_report.group(1)}"

        # Find output video path if visualization was enabled and successful
        if ENABLE_VISUALIZATION: # Check global config
            match_video = re.search(r"Output video segment: '(.+?)'", final_result_package) or \
                          re.search(r"Output video: '(.+?)'", final_result_package)
            if match_video:
                found_vid_path = match_video.group(1)
                if os.path.exists(found_vid_path): # Check if file really exists for display
                    output_video_display_path = found_vid_path
                else:
                    debug_print(f"Gradio: Video path found in log but file missing: {found_vid_path}")

    status_message_final = "Finished." if "✅" in final_result_package else "Finished with errors or warnings."

    yield {
        completion_text: gr.update(value=output_log_display),
        video_output: gr.update(value=output_video_display_path), # Update video component
        stat_frames: gr.update(value=frames_processed),
        stat_time: gr.update(value=time_seconds),
        stat_fps: gr.update(value=fps),
        stat_completed: gr.update(value=completed_paths),
        process_status_text: gr.update(value=status_message_final)
    }


# --- Gradio UI Definition ---
def create_interface():
    if not PROCESSOR_INITIALIZED:
        with gr.Blocks(title="Error") as interface_error:
            gr.Markdown("## FATAL ERROR: VideoProcessor failed to initialize. Application cannot start. Please check console logs.")
        return interface_error

    print("Creating Gradio interface...")
    theme = gr.themes.Soft(primary_hue=gr.themes.colors.emerald)
    css = """ footer { display: none !important; } """

    with gr.Blocks(title="Emaan Traffic Vehicle Detector", theme=theme, css=css) as interface:
        # --- Header ---
        viz_status_str = "Disabled"
        if ENABLE_VISUALIZATION:
            viz_status_str = "GPU (Experimental)" if GPU_VIZ_ENABLED else "CPU (Fallback)"

        gr.Markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 2.5em; font-weight: 600; color: #10b981;">🚦 Emaan Traffic Vehicle Detector</h1>
                <p style="font-size: 1.1em; color: #4b5563;">Upload video, configure, and process for vehicle tracking and counting.</p>
                <p style="font-size: 0.9em; color: #6b7280;">
                    <strong>Mode:</strong> `{PROCESSING_MODE.upper()}` |
                    <strong>Zone Def:</strong> `{LINE_MODE.upper()}` |
                    <strong>Output Video:</strong> `{viz_status_str}`
                </p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            # --- Tab 1: Setup ---
            with gr.TabItem("1️⃣ Setup", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Video Source")
                        video_input = gr.Video(label="Video File", height=360) # Keep type default
                    with gr.Column(scale=1):
                        gr.Markdown("### Initial Timestamp & Direction")
                        with gr.Group():
                             with gr.Row():
                                 start_date_input = gr.Textbox(label="Start Date (YYMMDD)", value=START_DATE)
                                 start_time_input = gr.Textbox(label="Start Time (HHMMSSmmm)", value=START_TIME)
                        primary_direction_input = gr.Dropdown(
                            label="Primary Direction for Report",
                            choices=["North", "South", "East", "West"], value="South"
                        )
                        status_text_draw = gr.Textbox( # For polygon drawing feedback
                            label="✏️ Drawing Helper",
                            value="Upload video first." if LINE_MODE == 'interactive' else "Polygon drawing disabled (hardcoded zones).",
                            interactive=False, lines=2,
                            visible=(LINE_MODE == 'interactive')
                        )

            # --- Tab 2: Draw Zones (Conditional, Keep as is) ---
            with gr.TabItem("✍️ Draw Zones", id=1, visible=(LINE_MODE == 'interactive')):
                gr.Markdown("### Define Detection Zones (Polygons)")
                gr.Markdown(
                     "Select direction, click vertices on image (min 3). Click 'Finish Current Polygon' when done for that zone."
                )
                with gr.Row(equal_height=False):
                     with gr.Column(scale=1, min_width=150):
                         gr.Markdown("**Select Direction & Actions:**")
                         north_btn = gr.Button("⬆️ North Zone"); south_btn = gr.Button("⬇️ South Zone")
                         east_btn = gr.Button("➡️ East Zone"); west_btn = gr.Button("⬅️ West Zone")
                         gr.Markdown("---")
                         finish_poly_btn = gr.Button("✅ Finish Current Polygon", variant="primary")
                         undo_point_btn = gr.Button("↩️ Undo Last Point")
                         clear_all_btn = gr.Button("❌ Clear All Polygons", variant="stop")
                     with gr.Column(scale=3):
                         image_display = gr.Image(label="Click to Define Polygon Vertices", type="numpy", interactive=True, height=450)

            # --- Tab 3: Process & Results ---
            with gr.TabItem("🚀 Process & View Results", id=2):
                gr.Markdown("### Start Processing")
                process_btn = gr.Button("📊 Process Video Now", variant="primary", scale=1)
                process_status_text = gr.Textbox(label="⚙️ Processing Status", value="Idle", interactive=False, lines=3, show_label=True) # Increased lines
                gr.Markdown("---")
                gr.Markdown("### Processing Output")
                with gr.Accordion("📊 Summary Statistics", open=True): # Open by default
                     with gr.Row(equal_height=True):
                         stat_frames = gr.Textbox(label="Frames Processed", value="N/A", interactive=False)
                         stat_time = gr.Textbox(label="Processing Time (s)", value="N/A", interactive=False)
                         stat_fps = gr.Textbox(label="Overall FPS", value="N/A", interactive=False)
                         stat_completed = gr.Textbox(label="Valid Paths", value="N/A", interactive=False)
                with gr.Group():
                    completion_text = gr.Textbox(label="📋 Processing Log & Report Paths", lines=8, interactive=False, show_copy_button=True) # Increased lines
                    video_output = gr.Video(label="🎬 Processed Video Output (if enabled & successful)", interactive=False)

        # --- Event Listeners ---
        # Setup Tab drawing outputs
        drawing_outputs = []
        if LINE_MODE == 'interactive':
            drawing_outputs = [image_display, status_text_draw]

        if PROCESSOR_INITIALIZED: # Only wire if processor is okay
            video_input.change(fn=handle_upload, inputs=[video_input], outputs=drawing_outputs, show_progress="minimal")

            if LINE_MODE == 'interactive':
                north_btn.click(lambda: handle_set_direction("north"), outputs=drawing_outputs, show_progress="minimal")
                south_btn.click(lambda: handle_set_direction("south"), outputs=drawing_outputs, show_progress="minimal")
                east_btn.click(lambda: handle_set_direction("east"), outputs=drawing_outputs, show_progress="minimal")
                west_btn.click(lambda: handle_set_direction("west"), outputs=drawing_outputs, show_progress="minimal")
                image_display.select(fn=handle_click, outputs=drawing_outputs, show_progress="minimal")
                finish_poly_btn.click(fn=handle_finish_polygon, outputs=drawing_outputs, show_progress="minimal")
                undo_point_btn.click(fn=handle_undo_point, outputs=drawing_outputs, show_progress="minimal")
                clear_all_btn.click(fn=handle_clear_polygons, outputs=drawing_outputs, show_progress="minimal")

            # Process Button outputs
            process_outputs_list = [
                completion_text, video_output,
                stat_frames, stat_time, stat_fps, stat_completed,
                process_status_text # Add new status text component here
            ]
            process_btn.click(
                fn=handle_process,
                inputs=[video_input, start_date_input, start_time_input, primary_direction_input],
                outputs=process_outputs_list # This expects a list of components
            )

    print("Gradio interface definition complete.")
    return interface