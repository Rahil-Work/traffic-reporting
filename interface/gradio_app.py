# project/interface/gradio_app.py
# This file is now only loaded and used if LINE_MODE == 'interactive'

import gradio as gr
import os
import re
from config import (
    START_DATE, START_TIME, PROCESSING_MODE, VIDEO_OUTPUT_DIR, LINE_MODE
)
from core.video_processor import VideoProcessor
from core.utils import debug_print

# --- Global Instance ---
print("Initializing VideoProcessor for Gradio (interactive mode)...")
try:
    processor = VideoProcessor()
    print(f"VideoProcessor Initialized (Line Mode: '{processor.line_mode}').")
    if processor.line_mode != 'interactive':
        print("WARNING: Gradio app loaded, but LINE_MODE is not 'interactive'. UI might not function as expected.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize VideoProcessor for Gradio. Error: {e}")
    import sys
    sys.exit(1)

# --- Gradio Event Handlers (Functionality Unchanged) ---
# [Keep handle_upload, handle_set_direction, handle_click, handle_process exactly as before]
def handle_upload(video_path):
    debug_print(f"Gradio: Handling video upload: {video_path}")
    # processor method checks mode internally, but we expect interactive here
    return processor.process_video_upload_for_gradio(video_path)

def handle_set_direction(direction):
    debug_print(f"Gradio: Setting direction to {direction}")
    return processor.set_gradio_direction(direction)

def handle_click(evt: gr.SelectData):
    debug_print(f"Gradio: Handling click event at index: {evt.index}")
    return processor.handle_gradio_click(evt)

def handle_process(video_path, start_date, start_time, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Starting Processing...")
    debug_print(f"Gradio: Starting video processing for {video_path}")
    if video_path is None:
        # Return updates for all output components on error
        return "Please upload a video first.", None, "N/A", "N/A", "N/A", "N/A"

    # Run the processing
    result_message_raw = processor.process_video(video_path, start_date, start_time)
    debug_print(f"Gradio: Processing finished.")

    # --- Parse Stats from Result Message ---
    frames_processed = "N/A"
    time_seconds = "N/A"
    fps = "N/A"
    completed_paths = "N/A"
    output_log = result_message_raw # Default log is the raw message

    if "✅ Processing completed" in result_message_raw:
        progress(1.0, desc="Processing Complete!")
        # Extract stats using regex
        match_frames = re.search(r"STAT_FRAMES_PROCESSED=(\d+)", result_message_raw)
        match_time = re.search(r"STAT_TIME_SECONDS=([\d\.]+)", result_message_raw)
        match_fps = re.search(r"STAT_FPS=([\d\.]+)", result_message_raw)
        match_paths = re.search(r"STAT_COMPLETED_PATHS=(\d+)", result_message_raw)

        if match_frames: frames_processed = match_frames.group(1)
        if match_time: time_seconds = match_time.group(1)
        if match_fps: fps = match_fps.group(1)
        if match_paths: completed_paths = match_paths.group(1)

        # Clean the log message by removing the STAT lines
        output_log = re.sub(r"--- Summary Stats ---.*", "", result_message_raw, flags=re.DOTALL).strip()
    else:
        # Handle processing error message
        output_log = result_message_raw # Show the error message
    # --- End Stat Parsing ---


    # Find output video path (same logic as before)
    output_video_path = None
    if "Processing completed" in result_message_raw:
        try:
            output_dir = VIDEO_OUTPUT_DIR
            files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("output_") and (f.endswith(".mp4") or f.endswith(".avi"))]
            if files: output_video_path = max(files, key=os.path.getmtime)
        except Exception as e: print(f"Error finding output video: {e}")

    # Return values for all output components
    return output_log, output_video_path, frames_processed, time_seconds, fps, completed_paths

# --- Gradio UI Definition ---
def create_interface():
    print("Creating Gradio interface...")

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
    .st-emotion-cache-18l72d6 { padding-top: 1rem; padding-bottom: 1rem; } /* Adjust group padding */
    footer { display: none !important; } /* Hide default Gradio footer */
    """

    with gr.Blocks(title="Emaan Traffic Vehicle Detector", theme=theme, css=css) as interface:
        # --- Header ---
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="font-size: 2.5em; font-weight: 600; color: #10b981;">🚦 Emaan Traffic Vehicle Detector</h1>
                <p style="font-size: 1.1em; color: #4b5563;">Upload a video, configure settings, and process to count vehicles.</p>
                <p style="font-size: 0.9em; color: #6b7280;"><strong>Mode:</strong> `{}` | <strong>Line Def:</strong> `{}`</p>
            </div>
            """.format(PROCESSING_MODE.upper(), LINE_MODE.upper())
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
                                     label="Start Date",
                                     value=START_DATE,
                                     placeholder="YYMMDD",
                                     # info="Format: YYMMDD" # Use info for hints
                                 )
                                 start_time = gr.Textbox(
                                     label="Start Time",
                                     value=START_TIME,
                                     placeholder="HHMMSSmmm",
                                     # info="Format: HHMMSSmmm (milliseconds optional)"
                                 )
                        gr.Markdown("---") # Separator
                        # Moved status text related to drawing here, makes more sense in interactive mode
                        status_text_draw = gr.Textbox(
                            label="✏️ Drawing Helper",
                            value="Upload video first." if LINE_MODE == 'interactive' else "Line drawing disabled (hardcoded lines).",
                            interactive=False,
                            lines=2,
                            visible=(LINE_MODE == 'interactive') # Only show if needed
                        )
                        gr.Markdown(
                            "➡️ Proceed to **Draw Lines** (if applicable) or **Process** tab.",
                            visible=(LINE_MODE == 'interactive')
                        )
                        gr.Markdown(
                            "➡️ Proceed to the **Process** tab.",
                            visible=(LINE_MODE != 'interactive')
                        )


            # --- Tab 2: Draw Lines (Conditional) ---
            with gr.TabItem("✍️ Draw Lines", id=1, visible=(LINE_MODE == 'interactive')):
                 gr.Markdown("### Define Detection Lines (Interactive)")
                 gr.Markdown("Select a direction, then click **two points** on the image below to draw the line for that direction. Repeat for all 4 directions.")
                 with gr.Row(equal_height=False):
                     with gr.Column(scale=1, min_width=150): # Buttons column
                         gr.Markdown("**Select Direction:**")
                         north_btn = gr.Button("⬆️ North", variant="secondary")
                         south_btn = gr.Button("⬇️ South", variant="secondary")
                         east_btn = gr.Button("➡️ East", variant="secondary")
                         west_btn = gr.Button("⬅️ West", variant="secondary")
                         # Re-use the drawing status text from the Setup tab for feedback
                         # (We'll update it via the handlers)
                     with gr.Column(scale=3): # Image column
                         image_display = gr.Image(
                             label="Click Image to Draw Lines",
                             type="numpy",
                             interactive=True,
                             height=450 # Make drawing area prominent
                         )


            # --- Tab 3: Process & Results ---
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
                    gr.Markdown("")
                    video_output = gr.Video(label="🎬 Processed Video", interactive=False)

        # --- Event Listeners (Connecting UI to Handlers) ---

        # Upload Action: Updates image display and drawing status text
        upload_outputs = []
        if LINE_MODE == 'interactive':
            upload_outputs.extend([image_display, status_text_draw]) # status_text_draw now exists
        video_input.change(
            fn=handle_upload,
            inputs=[video_input],
            outputs=upload_outputs, # Update necessary components
            show_progress="minimal"
        )

        if LINE_MODE == 'interactive':
            # Direction Button Actions: Update image display and drawing status text
            direction_outputs = [image_display, status_text_draw]
            north_btn.click(lambda: handle_set_direction("North"), outputs=direction_outputs, show_progress="minimal")
            south_btn.click(lambda: handle_set_direction("South"), outputs=direction_outputs, show_progress="minimal")
            east_btn.click(lambda: handle_set_direction("East"), outputs=direction_outputs, show_progress="minimal")
            west_btn.click(lambda: handle_set_direction("West"), outputs=direction_outputs, show_progress="minimal")

            # Image Click Action: Updates image display and drawing status text
            image_display.select(
                fn=handle_click,
                outputs=direction_outputs,
                show_progress="minimal"
            )

        # Process Button Action: Runs processing and updates all result components
        process_outputs = [
            completion_text, video_output,
            stat_frames, stat_time, stat_fps, stat_completed
        ]
        process_btn.click(
            fn=handle_process,
            inputs=[video_input, start_date, start_time],
            outputs=process_outputs,
            show_progress="full" # Use Gradio's built-in progress bar
        )

    print("Gradio interface created.")
    return interface