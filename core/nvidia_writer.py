# core/nvidia_writer.py
import PyNvVideoCodec as nvc # Using the correct import based on your feedback
import threading
import queue
import subprocess
import os
import time
# import numpy as np # May not be needed if direct GPU tensor works
# import torch # Only if you need to explicitly check/handle torch tensors

FFMPEG_PATH = "ffmpeg"
RAW_STREAM_FILENAME_SUFFIX = "raw_stream.bin"

class NvidiaFrameWriter:
    def __init__(self, output_path, width, height, fps,
                 encoder_codec='h264', # 'h264', 'hevc', 'av1' (for PyNvVideoCodec encoder)
                 bitrate='5M',
                 preset='P4', # PyNvVideoCodec doesn't seem to use Nvenc presets P1-P7 directly,
                              # but has 'tuning_info'. Let's assume you'll pass preset via kwargs
                              # or map it to 'tuning_info' and other params.
                 # CRITICAL: Format of the GPU TENSOR you will provide to put()
                 # e.g., "ARBG" if your tensor is uint8 RGBA (alpha first)
                 # e.g., "ABGR" if your tensor is uint8 BGRA (alpha first)
                 input_tensor_format_str="ARBG", # Or "ABGR", "NV12", "YUV420" etc.
                 temp_dir="temp_video_out", device_id=0): # device_id for encoder, gpuid in CreateEncoder is unused

        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps) if fps > 0 else 30
        
        self.ffmpeg_codec_name = encoder_codec.lower() # For FFmpeg muxing: 'h264', 'hevc'
        # For PyNvVideoCodec encoder, it takes kwargs like codec='h264'

        self.encoder_kwargs = { # Optional parameters for CreateEncoder
            'codec': encoder_codec, # 'h264', 'hevc', 'av1'
            'bitrate': int(bitrate.upper().replace('M', '000000').replace('K', '000')), # Convert to int
            'fps': self.fps,
            # 'preset': preset, # This specific 'preset' P1-P7 might not be a direct kwarg.
                               # Refer to Table 1 for available kwargs like 'tuning_info', 'rc'
            # Example:
            # 'tuning_info': 'high_quality', # or 'low_latency' etc.
            # 'rc': 'cbr', # 'cbr', 'vbr', 'constqp'
        }
        if preset: # Simple mapping, you might need more sophisticated logic
            if preset.lower() in ["p1", "p2", "p3"]:
                self.encoder_kwargs.setdefault('tuning_info', 'low_latency')
            elif preset.lower() in ["p4", "p5"]:
                self.encoder_kwargs.setdefault('tuning_info', 'high_quality') # Defaultish
            elif preset.lower() in ["p6", "p7"]:
                self.encoder_kwargs.setdefault('tuning_info', 'high_quality') # higher quality focus
            # Add more specific preset mappings to individual kwargs if needed

        self.input_tensor_format_str = input_tensor_format_str.upper() # e.g. "ARBG"
        
        self.temp_dir = str(temp_dir)
        base_out_name = os.path.splitext(os.path.basename(output_path))[0]
        self.raw_stream_path = os.path.join(self.temp_dir, f"{base_out_name}_{self.ffmpeg_codec_name}_{RAW_STREAM_FILENAME_SUFFIX}")
        
        self.device_id = device_id # Used for PyTorch tensor device assertion

        self.write_queue = queue.Queue(maxsize=int(self.fps * 2))
        self.stop_event = threading.Event()
        self.encoder = None
        self.thread = None
        self.packets_written_to_file = 0
        self.muxing_successful = False

        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"NvidiaWriter: Initializing for {self.output_path} ({self.width}x{self.height} @ {self.fps}fps)")
        print(f"NvidiaWriter: Input Tensor Format Expected by writer: {self.input_tensor_format_str}")
        print(f"NvidiaWriter: Encoder kwargs for CreateEncoder: {self.encoder_kwargs}")
        print(f"NvidiaWriter: Raw stream -> {self.raw_stream_path}")

    def _initialize_encoder(self):
        try:
            # usecpuinutbuffer = False means input tensor must be on GPU
            self.encoder = nvc.CreateEncoder(self.width, self.height,
                                             self.input_tensor_format_str, # "ARBG", "ABGR", "NV12" etc.
                                             usecpuinutbuffer=False,
                                             **self.encoder_kwargs)
            
            if self.encoder is None:
                print("NvidiaWriter: FATAL - nvc.CreateEncoder returned None. Check format/codec combination or resource limits.")
                return False # Explicitly return False if encoder is None
            
            print(f"NvidiaWriter: PyNvVideoCodec.CreateEncoder successful with format '{self.input_tensor_format_str}'.")
            return True
        except Exception as e:
            print(f"NvidiaWriter: FATAL - Error initializing PyNvVideoCodec.CreateEncoder:")
            print(f"  Width={self.width}, Height={self.height}, Format='{self.input_tensor_format_str}', usecpuinutbuffer=False")
            print(f"  Encoder Kwargs: {self.encoder_kwargs}")
            print(f"  Error: {e}")
            print(f"  Common issues: ")
            print(f"    - Unsupported format string '{self.input_tensor_format_str}' (check docs for exact valid strings).")
            print(f"    - Mismatch between tensor data and specified format.")
            print(f"    - GPU OOM or driver issues.")
            import traceback
            traceback.print_exc()
            self.encoder = None
            return False

    def _process_frame(self, gpu_tensor):
        """
        Prepares and encodes the GPU tensor.
        gpu_tensor is assumed to implement __cuda_array_interface__ (e.g., PyTorch tensor, CuPy array).
        """
        # Tensor preparation (minimal if format and device are correct)
        try:
            # Example for PyTorch tensor, adapt if using CuPy or other types
            if hasattr(gpu_tensor, 'is_cuda') and not gpu_tensor.is_cuda:
                print(f"NvidiaWriter: Error - Input tensor is not on GPU. Required for usecpuinutbuffer=False.")
                return False
            # Device check assumes PyTorch tensor, self.device_id corresponds to CUDA device ordinal
            if hasattr(gpu_tensor, 'device') and hasattr(gpu_tensor.device, 'index') and gpu_tensor.device.index != self.device_id:
                 print(f"NvidiaWriter: Warning - Tensor on device {gpu_tensor.device.index} but writer expected {self.device_id}. "
                       "Ensuring encoder can access it or moving is needed.")
                 # If PyNvVideoCodec uses its own context or a specific one, direct access might still work.
                 # Otherwise, gpu_tensor = gpu_tensor.to(f'cuda:{self.device_id}') # This would be a D2D copy

            # The docs mention "NCHW Tensor with batch count as 1 (N=1) and channel count as 1 (C=1)"
            # for YUV. For ARBG (which is interleaved like HWC if flattened, or CHW if channels first),
            # this might be different.
            # If your gpu_tensor is [H, W, C] (e.g., for ARBG [H, W, 4]), PyTorch __cuda_array_interface__
            # will describe it. If it's [C, H, W] ([4, H, W]), that's also fine.
            # PyNvVideoCodec should interpret it based on the `format` string given to CreateEncoder.

            # No explicit conversion to PySurface seems needed based on the docs if passing tensor directly.
            # The encoder.Encode() method itself handles __cuda_array_interface__ or torch.Tensor.

            encoded_bitstream_list = self.encoder.Encode(gpu_tensor)
            # Encode() returns an "array of encoded bitstream" - likely a list of bytearrays or bytes objects.
            
            if encoded_bitstream_list is None: # Or could be an empty list if no output yet
                # print("NvidiaWriter: Encode returned None/empty list (buffering or error).")
                return True # Not necessarily a failure, encoder might be buffering

            return encoded_bitstream_list # Return the list of packets

        except Exception as enc_ex:
            print(f"NvidiaWriter: Error during encoder.Encode(gpu_tensor): {enc_ex}")
            print(f"  Tensor type: {type(gpu_tensor)}, shape: {hasattr(gpu_tensor, 'shape') and gpu_tensor.shape}, dtype: {hasattr(gpu_tensor, 'dtype') and gpu_tensor.dtype}")
            return None # Indicate error

    def _run_encoding(self):
        if not self._initialize_encoder():
            # If encoder initialization fails, the thread should simply exit.
            # The main part of your application that started this thread
            # should detect that the writer didn't properly start or process anything.
            # No need to put anything on self.write_queue as it's an internal queue.
            print("NvidiaWriter: _initialize_encoder failed, _run_encoding cannot proceed.")
            return

        frames_processed_in_loop = 0
        try:
            with open(self.raw_stream_path, "wb") as f_out:
                while True:
                    gpu_tensor = None
                    try:
                        gpu_tensor = self.write_queue.get(block=True, timeout=0.1)
                    except queue.Empty:
                        if self.stop_event.is_set() and self.write_queue.empty():
                            print("NvidiaWriter: Stop event set and queue empty.")
                            break
                        continue

                    if gpu_tensor is None: # EOS marker
                        print("NvidiaWriter: EOS received from queue.")
                        self.write_queue.task_done()
                        break 

                    frames_processed_in_loop += 1
                    encoded_packets = self._process_frame(gpu_tensor)

                    if encoded_packets is None: # Indicates an error in _process_frame
                        print(f"NvidiaWriter: Critical error processing frame {frames_processed_in_loop}. Stopping.")
                        self.stop_event.set() # Signal issues
                        self.write_queue.task_done()
                        break
                    
                    if isinstance(encoded_packets, bool) and encoded_packets is True: # _process_frame buffered
                        self.write_queue.task_done()
                        continue

                    # encoded_packets should be a list of bytearrays/bytes
                    for packet_data in encoded_packets:
                        if packet_data and len(packet_data) > 0:
                            f_out.write(packet_data)
                            self.packets_written_to_file += 1
                    
                    self.write_queue.task_done()

                # --- Flush Encoder ---
                print("NvidiaWriter: Flushing encoder (calling EndEncode)...")
                try:
                    # EndEncode() flushes and returns pending bitstream data
                    final_packets = self.encoder.EndEncode()
                    if final_packets:
                        for packet_data in final_packets:
                            if packet_data and len(packet_data) > 0:
                                f_out.write(packet_data)
                                self.packets_written_to_file += 1
                        print(f"NvidiaWriter: Wrote {len(final_packets)} packets during EndEncode.")
                    else:
                        print("NvidiaWriter: EndEncode returned no packets.")
                except Exception as e_flush:
                    print(f"NvidiaWriter: Error during EndEncode: {e_flush}")

        except Exception as loop_err:
            print(f"NvidiaWriter: FATAL ERROR in encoding loop: {loop_err}")
            import traceback; traceback.print_exc()
            self.stop_event.set() # Ensure other parts know about the failure
        finally:
            print(f"NvidiaWriter: Encoding loop finished. Frames processed: {frames_processed_in_loop}. Total packets written to file: {self.packets_written_to_file}")
            # Signal EOS to consumer if not already done by init failure
            if self.encoder is None and frames_processed_in_loop == 0: # Init failed
                pass # Already signaled in _run_encoding's initial check
            elif self.write_queue.empty() and not self.stop_event.is_set(): # Normal exit or EOS from queue
                 pass # Consumer should get None from queue eventually if not already
            # The encoder object should release resources when GC'd.

    def _run_muxing(self):
        if not os.path.exists(self.raw_stream_path) or self.packets_written_to_file == 0:
            print(f"NvidiaWriter: Raw stream file '{self.raw_stream_path}' not found or empty. Skipping muxing.")
            return False

        output_dir = os.path.dirname(self.output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        # Use the same codec name for FFmpeg input format if it's a raw stream type
        # e.g., 'h264', 'hevc'.
        ffmpeg_input_format = self.ffmpeg_codec_name

        mux_command = [
            FFMPEG_PATH, '-y',
            '-loglevel', 'error',
            '-f', ffmpeg_input_format, # Format of the raw input stream
            '-r', str(self.fps),
            '-i', self.raw_stream_path,
            '-c:v', 'copy',
            '-movflags', '+faststart',
            self.output_path
        ]
        # ... (rest of _run_muxing is largely the same as before, handling subprocess) ...
        print(f"NvidiaWriter: Running FFmpeg muxing: {' '.join(mux_command)}")
        try:
            process = subprocess.Popen(mux_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=120)
            if process.returncode == 0:
                print(f"NvidiaWriter: Muxing complete. Final file: {self.output_path}")
                self.muxing_successful = True
                return True
            else:
                print(f"ERROR: FFmpeg muxing failed (code {process.returncode}):")
                if stdout: print(f"  FFmpeg stdout: {stdout.decode(errors='ignore')}")
                if stderr: print(f"  FFmpeg stderr: {stderr.decode(errors='ignore')}")
                self.muxing_successful = False
                return False
        except FileNotFoundError:
            print(f"ERROR: FFmpeg command '{FFMPEG_PATH}' not found.")
        except subprocess.TimeoutExpired:
            print("ERROR: FFmpeg muxing timed out.")
            if 'process' in locals(): process.kill()
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during FFmpeg muxing: {e}")
        self.muxing_successful = False
        return False


    def start(self):
        if self.thread is not None and self.thread.is_alive():
            print("NvidiaWriter: Warning - Start called but thread is already running.")
            return

        self.stop_event.clear()
        self.packets_written_to_file = 0
        self.muxing_successful = False

        if os.path.exists(self.raw_stream_path):
            try: os.remove(self.raw_stream_path)
            except Exception as e: print(f"NvidiaWriter: Warning - Failed to remove old raw stream '{self.raw_stream_path}': {e}")

        self.thread = threading.Thread(target=self._run_encoding, name="NvidiaWriterThread", daemon=True)
        self.thread.start()
        print("NvidiaWriter: Encoding thread started.")

    def put(self, gpu_tensor): # gpu_tensor is expected to be the raw tensor
        if not self.thread or not self.thread.is_alive() or self.stop_event.is_set():
            return False
        try:
            self.write_queue.put(gpu_tensor, block=True, timeout=1.0)
            return True
        except queue.Full:
            print("NvidiaWriter: Warning - Write queue full. Frame dropped.")
            return False
        except Exception as e:
            print(f"NvidiaWriter: Error putting frame on queue: {e}")
            return False

    def stop(self, cleanup_raw_file=True):
        print("NvidiaWriter: Stop requested...")
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            print("NvidiaWriter: Sending EOS to encoding queue...")
            try: self.write_queue.put(None, block=False, timeout=0.5) # Non-blocking if possible
            except queue.Full: print("NvidiaWriter: Queue full sending EOS, thread will see stop_event.")
            except Exception: pass

            print("NvidiaWriter: Waiting for encoding thread to finish...")
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                print("NvidiaWriter: WARNING - Encoding thread join timed out.")
            else:
                print("NvidiaWriter: Encoding thread finished.")
        else:
            print("NvidiaWriter: Encoding thread was not running or already stopped.")

        mux_success = self._run_muxing()

        if cleanup_raw_file and os.path.exists(self.raw_stream_path):
            try: os.remove(self.raw_stream_path)
            except Exception as e: print(f"NvidiaWriter: Warning - Failed to delete raw stream '{self.raw_stream_path}': {e}")
        elif not cleanup_raw_file: print(f"NvidiaWriter: Raw stream file kept: {self.raw_stream_path}")

        self.thread = None
        return mux_success and self.muxing_successful