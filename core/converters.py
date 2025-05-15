# project/core/converters.py
import subprocess
import os
import tempfile
from config import FFMPEG_PATH

def convert_dav_to_mp4_internal(dav_filepath):
    """
    Converts a .dav file to .mp4 using ffmpeg (path determined by find_ffmpeg_path)
    Returns the path to the converted .mp4 file, or None on failure.
    The converted file is placed in the system's temporary directory.
    The caller is responsible for deleting this temporary output MP4 file.
    """
    if not FFMPEG_PATH: # Use the globally determined path
        print("ERROR: [convert_dav] FFmpeg executable path not determined. Cannot convert .dav file.")
        return None
    
    ffmpeg_executable = FFMPEG_PATH
    if not os.path.exists(dav_filepath):
        print(f"ERROR: [convert_dav] Input .dav file not found: {dav_filepath}")
        return None

    try:
        dav_basename = os.path.basename(dav_filepath)
        dav_name_no_ext = os.path.splitext(dav_basename)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', prefix=f"{dav_name_no_ext}_conv_") as tmpfile:
            output_mp4_filepath = tmpfile.name
        
    except Exception as e_tmp:
        print(f"ERROR: [convert_dav] Could not create temporary output file path: {e_tmp}")
        return None

    command = [
                ffmpeg_executable,
                "-y",                 # Overwrite output
                "-i", dav_filepath,   # Input
                "-an",                # No audio
                "-c:v", "copy",       # Copy video stream
                output_mp4_filepath       # Output
            ]

    print(f"INFO: [convert_dav] Attempting to convert '{dav_filepath}' to '{output_mp4_filepath}'...")
    print(f"DEBUG: [convert_dav] FFmpeg command: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')     
        if os.path.exists(output_mp4_filepath) and os.path.getsize(output_mp4_filepath) > 1024: 
            print(f"INFO: [convert_dav] Conversion successful: {output_mp4_filepath}")
            return output_mp4_filepath
        else:
            print(f"ERROR: [convert_dav] Conversion command ran, output file '{output_mp4_filepath}' missing or too small.")
            if process.stderr: print(f"DEBUG: [convert_dav] FFmpeg stderr:\n{process.stderr.strip()[:1000]}")
            return None
    except subprocess.CalledProcessError as e:
        stdout_short = (e.stdout or "")[:500] + ("..." if len(e.stdout or "") > 500 else "")
        stderr_short = (e.stderr or "")[:1000] + ("..." if len(e.stderr or "") > 1000 else "")
        print(f"ERROR: [convert_dav] FFmpeg failed for '{dav_filepath}':")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  FFmpeg STDOUT (truncated):\n{stdout_short.strip()}")
        print(f"  FFmpeg STDERR (truncated):\n{stderr_short.strip()}")
        if os.path.exists(output_mp4_filepath):
            try: os.remove(output_mp4_filepath)
            except OSError: pass
        return None
    except FileNotFoundError:
        print(f"ERROR: [convert_dav] FFmpeg command '{ffmpeg_executable}' not found during subprocess run.")
        return None
    except Exception as e_gen:
        print(f"ERROR: [convert_dav] An unexpected error occurred during DAV conversion: {e_gen}")
        import traceback
        traceback.print_exc()
        if os.path.exists(output_mp4_filepath):
            try: os.remove(output_mp4_filepath)
            except OSError: pass
        return None