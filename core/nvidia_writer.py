# core/nvidia_writer.py
import PyNvCodec as nvc
import threading
import queue
import subprocess
import os
from ..config import FFMPEG_PATH, RAW_STREAM_FILENAME # Import config vars

class NvidiaFrameWriter:
    def __init__(self, output_path, width, height, fps, codec, bitrate, preset, temp_dir, device_id=0):
        self.output_path = output_path # Final MP4/MKV path
        self.width = width
        self.height = height
        self.fps = int(fps) if fps > 0 else 30 # Ensure valid fps
        self.codec = codec
        self.bitrate = bitrate
        self.preset = preset
        self.temp_dir = temp_dir
        self.raw_stream_path = os.path.join(self.temp_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_{RAW_STREAM_FILENAME}")
        self.device_id = device_id
        self.write_queue = queue.Queue(maxsize=int(self.fps * 2)) # Buffer ~2 seconds
        self.stop_event = threading.Event()
        self.encoder = None
        self.thread = None
        self.packets_written = 0
        self.muxing_successful = False

        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"NvidiaWriter: Initializing for {output_path} ({width}x{height} @ {self.fps}fps)")
        print(f"NvidiaWriter: Codec={codec}, Bitrate={bitrate}, Preset={preset}")
        print(f"NvidiaWriter: Raw stream -> {self.raw_stream_path}")


    def _initialize_encoder(self):
        # Critical: Define the pixel format matching the TENSOR coming from gpu_overlay
        # Assuming gpu_overlay outputs RGB Float[0,1], but encoder might need uint8 RGB/BGR?
        # Check PyNvEncoder docs. Let's ASSUME it needs RGB planar uint8 for now.
        # This requires conversion before putting on queue or within the writer thread.
        # For simplicity, let's try passing RGB directly, might need fmt='rgb' or similar.
        input_pixel_format = nvc.PixelFormat.RGB # MATCH THIS TO YOUR TENSOR FORMAT

        settings = {
            'preset': self.preset, 'codec': self.codec,
            's': f'{self.width}x{self.height}', 'bitrate': self.bitrate,
            'fps': str(self.fps), 'fmt': input_pixel_format.name # Use enum name
        }
        try:
            self.encoder = nvc.PyNvEncoder(settings, self.device_id)
            print("NvidiaWriter: Encoder initialized successfully.")
            return True
        except Exception as e:
            print(f"NvidiaWriter: ERROR initializing encoder (check settings, esp. 'fmt'): {e}")
            self.encoder = None
            return False

    def _run_encoding(self):
        if not self._initialize_encoder(): return

        try:
            with open(self.raw_stream_path, "wb") as f_out:
                while True:
                    try: gpu_tensor = self.write_queue.get(timeout=5.0)
                    except queue.Empty:
                        if self.stop_event.is_set() and self.write_queue.empty(): break
                        continue

                    if gpu_tensor is None:
                        self.write_queue.task_done(); break # EOS

                    # --- Input Tensor Preparation (IF NEEDED) ---
                    # Example: If encoder needs uint8 [0, 255] but tensor is float [0, 1]
                    # if gpu_tensor.dtype == torch.float32 or gpu_tensor.dtype == torch.float16:
                    #     gpu_tensor = (gpu_tensor * 255.0).byte()
                    # Ensure contiguous and correct device
                    if not gpu_tensor.is_contiguous(): gpu_tensor = gpu_tensor.contiguous()
                    if gpu_tensor.device.index != self.device_id: gpu_tensor = gpu_tensor.to(f'cuda:{self.device_id}')
                    # --- Encode ---
                    try:
                         # Try EncodeSingleTensor first (more direct)
                         success = self.encoder.EncodeSingleTensor(gpu_tensor)
                         if not success: print("NvidiaWriter: EncodeSingleTensor failed.")
                    except AttributeError:
                         print("NvidiaWriter: CRITICAL - EncodeSingleTensor not available. Implement Surface conversion.")
                         break # Cannot proceed
                    except Exception as enc_ex:
                         print(f"NvidiaWriter: Error during EncodeSingleTensor: {enc_ex}")
                         # Maybe try skipping frame? For now, break on error.
                         break

                    self.write_queue.task_done()

                    # --- Retrieve and Write Packets ---
                    encoded_packet = bytearray()
                    while True:
                        success = self.encoder.GetEncodedPacket(encoded_packet)
                        if not success: break
                        f_out.write(encoded_packet)
                        self.packets_written += 1

                # --- Flush Encoder ---
                print("NvidiaWriter: Flushing encoder...")
                if self.encoder.Flush(): # Check return value if possible
                     encoded_packet = bytearray()
                     while self.encoder.GetEncodedPacket(encoded_packet):
                          f_out.write(encoded_packet); self.packets_written += 1
                else: print("NvidiaWriter: Warning - Flush command potentially failed.")

        except Exception as loop_err:
             print(f"NvidiaWriter: ERROR in encoding loop: {loop_err}")
             import traceback; traceback.print_exc()
        finally:
            print(f"NvidiaWriter: Encoding loop finished. Wrote {self.packets_written} packets to {self.raw_stream_path}")

    def _run_muxing(self):
         if not os.path.exists(self.raw_stream_path) or self.packets_written == 0:
              print("NvidiaWriter: Raw stream file not found or empty. Skipping muxing.")
              return False
         os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
         mux_command = [
             FFMPEG_PATH, '-y', '-loglevel', 'error',
             '-f', self.codec, '-r', str(self.fps), '-i', self.raw_stream_path,
             '-c:v', 'copy', self.output_path
         ]
         print(f"NvidiaWriter: Running FFmpeg muxing: {' '.join(mux_command)}")
         try:
              subprocess.run(mux_command, check=True, capture_output=True, text=True)
              print(f"NvidiaWriter: Muxing complete. Final file: {self.output_path}")
              self.muxing_successful = True
              return True
         except FileNotFoundError: print(f"ERROR: '{FFMPEG_PATH}' command not found during muxing.")
         except subprocess.CalledProcessError as e: print(f"ERROR: FFmpeg muxing failed: {e.stderr}")
         except Exception as e: print(f"ERROR: An unexpected error occurred during FFmpeg muxing: {e}")
         self.muxing_successful = False
         return False

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.packets_written = 0
            self.muxing_successful = False
            self.thread = threading.Thread(target=self._run_encoding, daemon=True)
            self.thread.start()
            print("NvidiaWriter: Encoding thread started.")

    def put(self, gpu_tensor):
        if self.thread and self.thread.is_alive():
             try: self.write_queue.put(gpu_tensor, block=True, timeout=1.0) # Block with timeout
             except queue.Full: print("NvidiaWriter: Warning - Write queue full. Frame dropped.")

    def stop(self):
        print("NvidiaWriter: Stopping...")
        if self.thread and self.thread.is_alive():
             self.stop_event.set()
             try: self.write_queue.put(None, timeout=1.0) # Send EOS
             except queue.Full: pass # If full, thread should eventually see stop_event
             self.thread.join(timeout=20) # Increase timeout?
             if self.thread.is_alive(): print("NvidiaWriter: Warning - Encoding thread join timed out.")
             else: print("NvidiaWriter: Encoding thread finished.")
        else: print("NvidiaWriter: Thread not running.")

        mux_success = self._run_muxing() # Run muxing after thread stops

        # Cleanup raw stream file
        if os.path.exists(self.raw_stream_path):
             try: os.remove(self.raw_stream_path)
             except Exception as e: print(f"NvidiaWriter: Warning - Failed to delete raw stream file: {e}")

        self.thread = None
        return mux_success # Return if muxing was ok