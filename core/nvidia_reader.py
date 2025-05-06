# core/nvidia_reader.py
import PyNvCodec as nvc
import queue

class NvidiaFrameReader:
    def __init__(self, video_path, target_fps, original_fps, frame_queue, stop_event, device_id=0):
        self.video_path = str(video_path)
        self.target_fps = target_fps
        self.original_fps = original_fps if original_fps > 0 else 30.0
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.device_id = device_id

        self.frame_interval = max(1.0, self.original_fps / self.target_fps)
        self.next_frame_idx_to_yield = 0.0
        self.total_frames_yielded = 0
        self.decoder = None
        self.cc = None
        # Target format for model input (RGB is common)
        self.target_format = nvc.PixelFormat.RGB
        self.width = 0
        self.height = 0

        print(f"NvidiaReader: Initializing for {self.video_path}")
        print(f"NvidiaReader: Target FPS={self.target_fps}, Orig FPS={self.original_fps}, Interval={self.frame_interval:.2f}")

    def _initialize_decoder(self):
        try:
            self.decoder = nvc.PyNvDecoder(self.video_path, self.device_id)
            self.width = self.decoder.Width()
            self.height = self.decoder.Height()
            native_format = self.decoder.Format()

            print(f"NvidiaReader: Decoder initialized. Dimensions: {self.width}x{self.height}, Native Format: {native_format}")

            if native_format != self.target_format:
                self.cc = nvc.PySurfaceConverter(self.width, self.height, native_format, self.target_format, self.device_id)
                print(f"NvidiaReader: Color Converter {native_format} -> {self.target_format} initialized.")
            else:
                self.cc = None
                print("NvidiaReader: No color conversion needed.")
            return True
        except Exception as e:
            print(f"NvidiaReader: ERROR initializing decoder or converter: {e}")
            self.decoder = None
            self.cc = None
            return False

    def run(self):
        if not self._initialize_decoder():
            print("NvidiaReader: Initialization failed. Exiting thread.")
            self.frame_queue.put(None)
            return

        decoded_frame_count = 0

        try:
            while not self.stop_event.is_set():
                gpu_surface = self.decoder.DecodeSingleSurface()
                if gpu_surface is None or gpu_surface.Empty():
                    print("NvidiaReader: Decode finished or failed (EOS or Error).")
                    break

                decoded_frame_count += 1

                if decoded_frame_count >= self.next_frame_idx_to_yield:
                    frame_time_sec = decoded_frame_count / self.original_fps
                    target_surface = self.cc.Execute(gpu_surface) if self.cc else gpu_surface
                    if target_surface is None or target_surface.Empty():
                        print(f"NvidiaReader: Warning - Color conversion failed for frame {decoded_frame_count}. Skipping.")
                        continue

                    try:
                        # Convert surface to planar RGB Tensor [3, H, W] on GPU
                        frame_tensor = nvc.SurfaceToNvTensor(target_surface)
                    except AttributeError:
                        print("NvidiaReader: CRITICAL - SurfaceToNvTensor not available. Install compatible PyNvCodec/drivers or implement fallback.")
                        break # Cannot proceed efficiently
                    except Exception as tensor_ex:
                        print(f"NvidiaReader: Error converting surface to tensor for frame {decoded_frame_count}: {tensor_ex}")
                        continue

                    # Enqueue the GPU tensor, frame count, and timestamp
                    try:
                        self.frame_queue.put((frame_tensor, decoded_frame_count, frame_time_sec), block=True, timeout=5.0)
                        self.total_frames_yielded += 1
                        self.next_frame_idx_to_yield += self.frame_interval
                    except queue.Full:
                        print("NvidiaReader: Queue full, waiting...")
                        if self.stop_event.wait(0.1): break
                        try: # Retry
                            self.frame_queue.put((frame_tensor, decoded_frame_count, frame_time_sec), block=True, timeout=5.0)
                            self.total_frames_yielded += 1
                            self.next_frame_idx_to_yield += self.frame_interval
                        except queue.Full:
                            print("NvidiaReader: ERROR - Queue persistently full. Stopping thread.")
                            break
                    except Exception as q_err:
                         print(f"NvidiaReader: Error putting frame on queue: {q_err}")
                         break

        except Exception as loop_err:
            print(f"NvidiaReader: ERROR during decode loop: {loop_err}")
            import traceback; traceback.print_exc()
        finally:
            print(f"NvidiaReader: Exiting. Decoded {decoded_frame_count} frames, Yielded {self.total_frames_yielded} frames.")
            self.frame_queue.put(None) # Signal end