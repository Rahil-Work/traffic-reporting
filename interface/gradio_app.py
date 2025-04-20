# project/interface/gradio_app.py
# This file is now only loaded and used if LINE_MODE == 'interactive'

import gradio as gr
import os
import re
# Import cv2 and numpy if needed for any direct drawing in handlers,
# though ideally drawing happens within VideoProcessor methods.
# import cv2
# import numpy as np
from config import (
    START_DATE, START_TIME, PROCESSING_MODE, VIDEO_OUTPUT_DIR, LINE_MODE
)
from core.video_processor import VideoProcessor
from core.utils import debug_print

# --- Global Instance ---
# Initialize processor as before
print("Initializing VideoProcessor for Gradio (interactive mode)...")
try:
    processor = VideoProcessor() # Assumes VideoProcessor is updated for polygons
    print(f"VideoProcessor Initialized (Line Mode: '{processor.line_mode}').")
    if processor.line_mode != 'interactive':
        print("WARNING: Gradio app loaded, but LINE_MODE is not 'interactive'. UI might not function as expected.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize VideoProcessor for Gradio. Error: {e}")
    import sys
    sys.exit(1)

# --- Gradio Event Handlers ---

# Handle video upload - gets first frame for drawing
def handle_upload(video_path):
    debug_print(f"Gradio: Handling video upload: {video_path}")
    # Processor method should now reset polygon state and prepare frame
    return processor.process_video_upload_for_gradio(video_path) # Returns initial frame, status

# Set the direction for drawing the *next* polygon
def handle_set_direction(direction):
    debug_print(f"Gradio: Setting direction for polygon: {direction}")
    # Processor method should update current_gradio_direction and clear temp points
    frame_with_polys, status = processor.set_gradio_polygon_direction(direction)
    return frame_with_polys, status

# Handle clicks on the image to add polygon vertices
def handle_click(evt: gr.SelectData):
    debug_print(f"Gradio: Handling click event at index: {evt.index} for polygon vertex")
    # Processor method should add point to temp points and redraw the frame
    frame_with_polys, status = processor.handle_gradio_polygon_click(evt)
    return frame_with_polys, status

# Finalize the currently drawn polygon points
def handle_finish_polygon():
    debug_print(f"Gradio: Finishing current polygon")
    # Processor method should validate temp points, add to main polygon dict, clear temp points, redraw
    frame_with_polys, status = processor.finalize_current_polygon()
    return frame_with_polys, status

# Undo the last point added
def handle_undo_point():
    debug_print(f"Gradio: Undoing last point")
    # Call the method in VideoProcessor to handle the undo logic
    frame_with_polys, status = processor.undo_last_polygon_point()
    return frame_with_polys, status

# Clear all drawn polygons (optional reset)
def handle_clear_polygons():
    debug_print(f"Gradio: Clearing all polygons")
    frame_with_polys, status = processor.reset_gradio_polygons()
    return frame_with_polys, status

# Process the video (uses finalized polygons)
def handle_process(video_path, start_date, start_time, progress=gr.Progress(track_tqdm=True)):
    # This function structure remains the same as before
    # It calls processor.process_video, which internally uses the finalized polygons
    # if LINE_MODE is interactive.
    progress(0, desc="Starting Processing...")
    debug_print(f"Gradio: Starting video processing for {video_path}")
    if video_path is None:
        # Return updates for all output components on error
        return "Please upload a video first.", None, "N/A", "N/A", "N/A", "N/A"

    # Run the processing - processor.process_video uses self.gradio_polygons
    # Add a try-except block here to catch potential errors during processing setup (like zone validation)
    # and report them back to the Gradio UI gracefully.
    try:
        result_message_raw = processor.process_video(video_path, start_date, start_time)
        debug_print(f"Gradio: Processing finished.")
    except ValueError as ve:
         # Catch specific validation errors (like not enough zones)
         print(f"Gradio: Validation Error during processing setup: {ve}")
         progress(0, desc="Error!") # Update progress bar on error
         return f"Error: {ve}", None, "N/A", "N/A", "N/A", "N/A"
    except Exception as e:
         # Catch any other unexpected errors during processing
         print(f"Gradio: Unexpected Error during processing: {e}")
         import traceback; traceback.print_exc()
         progress(0, desc="Error!")
         return f"Unexpected Error: {e}", None, "N/A", "N/A", "N/A", "N/A"


    # --- Stat Parsing (No change needed here) ---
    frames_processed = "N/A"; time_seconds = "N/A"; fps = "N/A"; completed_paths = "N/A"
    output_log = result_message_raw
    if "✅ Processing completed" in result_message_raw:
        progress(1.0, desc="Processing Complete!") # Progress bar should complete here
        match_frames = re.search(r"STAT_FRAMES_PROCESSED=(\d+)", result_message_raw)
        match_time = re.search(r"STAT_TIME_SECONDS=([\d\.]+)", result_message_raw)
        match_fps = re.search(r"STAT_FPS=([\d\.]+)", result_message_raw)
        match_paths = re.search(r"STAT_COMPLETED_PATHS=(\d+)", result_message_raw)
        if match_frames: frames_processed = match_frames.group(1)
        if match_time: time_seconds = match_time.group(1)
        if match_fps: fps = match_fps.group(1)
        if match_paths: completed_paths = match_paths.group(1)
        output_log = re.sub(r"--- Summary Stats ---.*", "", result_message_raw, flags=re.DOTALL).strip()
    elif result_message_raw.startswith("❌ Error:") or result_message_raw.startswith("Error:"):
         # If process_video returned an error string directly
         output_log = result_message_raw
         progress(0, desc="Error!") # Ensure progress shows error if processing failed internally
    else:
        # Handle unexpected result format if necessary
        output_log = "Processing finished, but result format unexpected."
        progress(1.0, desc="Finished (Unknown State)")


    # --- Find Output Video (No change needed here) ---
    output_video_path = None
    if "✅ Processing completed" in result_message_raw:
        try:
            output_dir = VIDEO_OUTPUT_DIR
            files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("output_") and (f.endswith(".mp4") or f.endswith(".avi"))]
            if files: output_video_path = max(files, key=os.path.getmtime)
        except Exception as e: print(f"Error finding output video: {e}")

    # Return values for all output components
    return output_log, output_video_path, frames_processed, time_seconds, fps, completed_paths


# --- Gradio UI Definition ---
def create_interface():
    print("Creating Gradio interface (Polygon Mode)...")

    # Use a slightly more vibrant theme
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    # CSS for minor tweaks (optional)
    css = """
    .gradio-container { max-width: 1280px !important; margin: auto; }
    /* Add other CSS tweaks if desired */
    footer { display: none !important; } /* Hide default Gradio footer */
    """

    with gr.Blocks(title="Emaan Traffic Vehicle Detector", theme=theme, css=css) as interface:
        # --- Header ---
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 2.5em; font-weight: 600; color: #10b981;">🚦 Emaan Traffic Vehicle Detector</h1>
                <p style="font-size: 1.1em; color: #4b5563;">Upload a video, configure settings, and process to count vehicles.</p>
                <p style="font-size: 0.9em; color: #6b7280;"><strong>Mode:</strong> `{}` | <strong>Zone Def:</strong> `{}`</p>
            </div>
            """.format(PROCESSING_MODE.upper(), "POLYGONS" if LINE_MODE == 'interactive' else "HARDCODED") # Update text
        )

        # --- Main Layout with Tabs ---
        with gr.Tabs() as tabs:
            # --- Tab 1: Setup ---
            with gr.TabItem("1️⃣ Setup", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Video Source")
                        video_input = gr.Video(label="Video File", height=360)
                    with gr.Column(scale=1):
                        gr.Markdown("### Set Initial Timestamp")
                        gr.Markdown("Enter the date and time corresponding to the *start* of the video.")
                        with gr.Group(): # Group timestamp inputs
                             with gr.Row():
                                 start_date = gr.Textbox(
                                     label="Start Date", value=START_DATE, placeholder="YYMMDD"
                                 )
                                 start_time = gr.Textbox(
                                     label="Start Time", value=START_TIME, placeholder="HHMMSSmmm"
                                 )
                        gr.Markdown("---") # Separator
                        status_text_draw = gr.Textbox(
                            label="✏️ Drawing Helper",
                            value="Upload video first." if LINE_MODE == 'interactive' else "Polygon drawing disabled (hardcoded zones).",
                            interactive=False,
                            lines=3, # Keep slightly larger
                            visible=(LINE_MODE == 'interactive')
                        )
                        gr.Markdown(
                            "➡️ Proceed to **Draw Zones** tab (if interactive) or **Process** tab." if LINE_MODE == 'interactive' else "➡️ Proceed to the **Process** tab.",
                            visible=True
                        )


            # --- Tab 2: Draw Zones (Conditional) ---
            with gr.TabItem("✍️ Draw Zones", id=1, visible=(LINE_MODE == 'interactive')):
                 gr.Markdown("### Define Detection Zones (Polygons)")
                 # Updated Instructions
                 gr.Markdown(
                     "Select a direction, then **click the desired vertices** (minimum 3) on the image below to define the polygon zone. "
                     "You **do not** need to click the starting point again to close it. "
                     "Use 'Undo Last Point' to correct mistakes. Click **'Finish Current Polygon'** when all vertices for the current zone are placed."
                 )
                 with gr.Row(equal_height=False):
                     with gr.Column(scale=1, min_width=150): # Controls column
                         gr.Markdown("**Select Direction:**")
                         north_btn = gr.Button("⬆️ North Zone", variant="secondary")
                         south_btn = gr.Button("⬇️ South Zone", variant="secondary")
                         east_btn = gr.Button("➡️ East Zone", variant="secondary")
                         west_btn = gr.Button("⬅️ West Zone", variant="secondary")
                         gr.Markdown("---")
                         finish_poly_btn = gr.Button("✅ Finish Current Polygon", variant="primary")
                         undo_point_btn = gr.Button("↩️ Undo Last Point") # Added button
                         clear_all_btn = gr.Button("❌ Clear All Polygons", variant="stop")
                         # Reuse the status text from Setup tab for feedback
                     with gr.Column(scale=3): # Image column
                         image_display = gr.Image(
                             label="Click Image to Define Polygon Vertices",
                             type="numpy",
                             interactive=True, # Enable clicks
                             height=450 # Make drawing area prominent
                         )


            # --- Tab 3: Process & Results ---
            # (No structural changes needed in this tab's definition)
            with gr.TabItem("🚀 Process & View Results", id=2):
                gr.Markdown("### Start Processing")
                with gr.Row():
                    process_btn = gr.Button("📊 Process Video Now", variant="primary", scale=1)
                gr.Markdown("---")
                gr.Markdown("### Processing Output")
                with gr.Accordion("📊 Summary Statistics", open=False): # Collapsible stats
                     with gr.Row(equal_height=True):
                         stat_frames = gr.Textbox(label="Frames Processed", value="N/A", interactive=False, text_align="center")
                         stat_time = gr.Textbox(label="Processing Time (s)", value="N/A", interactive=False, text_align="center")
                         stat_fps = gr.Textbox(label="Overall FPS", value="N/A", interactive=False, text_align="center")
                         stat_completed = gr.Textbox(label="Valid Paths", value="N/A", interactive=False, text_align="center")

                with gr.Group(): # Group log and video output
                    completion_text = gr.Textbox(label="📋 Processing Log", lines=6, interactive=False, show_copy_button=True)
                    gr.Markdown("") # Spacer
                    video_output = gr.Video(label="🎬 Processed Video", interactive=False)

        # --- Event Listeners (Connecting UI to Handlers) ---

        # Define outputs commonly updated by drawing actions
        drawing_outputs = []
        if LINE_MODE == 'interactive':
            drawing_outputs = [image_display, status_text_draw]

        # Upload Action: Updates image display and drawing status text
        video_input.change(
            fn=handle_upload,
            inputs=[video_input],
            outputs=drawing_outputs, # Use defined list
            show_progress="minimal"
        )

        if LINE_MODE == 'interactive':
            # Direction Button Actions: Update image display and drawing status text
            north_btn.click(lambda: handle_set_direction("north"), outputs=drawing_outputs, show_progress="minimal")
            south_btn.click(lambda: handle_set_direction("south"), outputs=drawing_outputs, show_progress="minimal")
            east_btn.click(lambda: handle_set_direction("east"), outputs=drawing_outputs, show_progress="minimal")
            west_btn.click(lambda: handle_set_direction("west"), outputs=drawing_outputs, show_progress="minimal")

            # Image Click Action: Updates image display and drawing status text
            image_display.select(
                fn=handle_click,
                outputs=drawing_outputs,
                show_progress="minimal"
            )

            # Finish Polygon Button: Finalize points and update display/status
            finish_poly_btn.click(fn=handle_finish_polygon, outputs=drawing_outputs, show_progress="minimal")

            # Undo Point Button: Remove last point and update display/status
            undo_point_btn.click(fn=handle_undo_point, outputs=drawing_outputs, show_progress="minimal")

            # Clear All Button: Reset polygons and update display/status
            clear_all_btn.click(fn=handle_clear_polygons, outputs=drawing_outputs, show_progress="minimal")


        # Process Button Action: Runs processing and updates all result components
        process_outputs = [
            completion_text, video_output,
            stat_frames, stat_time, stat_fps, stat_completed
        ]
        # Assign the process_btn click handler
        process_btn.click(
            fn=handle_process,
            inputs=[video_input, start_date, start_time],
            outputs=process_outputs # All result components
        )

    print("Gradio interface created (Polygon Mode).")
    return interface