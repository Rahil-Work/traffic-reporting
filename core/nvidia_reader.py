# core/nvidia_reader.py
import PyNvVideoCodec as nvc # Assuming PyNvVideoCodec is the correct import name
import queue
import time
import torch # For torch.from_dlpack
import subprocess
import os # For os.getenv, os.path.exists, os.path.basename
import re # For parsing frame dimensions from string representation
import threading # For threading.Event

# --- Helper Function for Codec Detection (using ffprobe) ---
def get_video_codec_nvc_enum(video_path_str, nvc_module_ref):
    """
    Determines the video codec using ffprobe and returns the corresponding
    PyNvVideoCodec.cudaVideoCodec enum member.
    Args:
        video_path_str (str): Path to the video file.
        nvc_module_ref: The imported PyNvVideoCodec module (passed as 'nvc').
    Returns:
        PyNvVideoCodec.cudaVideoCodec enum member or None if detection fails.
    """
    # Use FFPROBE_PATH env var; defaults to 'ffprobe' assuming it's in system PATH.
    ffprobe_path = os.getenv("FFPROBE_PATH", "ffprobe")
    thread_prefix = f"NvidiaReader-ffprobe [{os.path.basename(video_path_str)}]:"

    try:
        if not os.path.exists(video_path_str):
            print(f"{thread_prefix} Video file not found at '{video_path_str}'.")
            return None

        command = [
            ffprobe_path,
            "-v", "error",             # Suppress verbose output, only show errors
            "-select_streams", "v:0",  # Select only the first video stream
            "-show_entries", "stream=codec_name", # Get the codec name
            "-of", "default=noprint_wrappers=1:nokey=1", # Output format: value only
            video_path_str
        ]
        # Increased timeout for ffprobe, especially for network files or slow disks
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        codec_name_str = result.stdout.strip().lower()

        print(f"{thread_prefix} ffprobe detected codec_name string: '{codec_name_str}'")

        if codec_name_str == "h264":
            return nvc_module_ref.cudaVideoCodec.H264
        elif codec_name_str == "hevc": # H.265
            return nvc_module_ref.cudaVideoCodec.HEVC
        elif codec_name_str == "av1":
            return nvc_module_ref.cudaVideoCodec.AV1
        # Add more mappings if PyNvVideoCodec supports them and your videos use them
        # e.g., vp9, mpeg2. Check nvc_module_ref.cudaVideoCodec for available enums.
        # elif codec_name_str == "vp9":
        #     if hasattr(nvc_module_ref.cudaVideoCodec, 'VP9'):
        #         return nvc_module_ref.cudaVideoCodec.VP9
        #     else:
        #         print(f"{thread_prefix} ffprobe detected 'vp9', but PyNvVideoCodec.cudaVideoCodec.VP9 not found.")
        #         return None
        else:
            print(f"{thread_prefix} Unknown or unsupported codec string from ffprobe: '{codec_name_str}'")
            return None
    except FileNotFoundError:
        print(f"{thread_prefix} CRITICAL - ffprobe command not found at '{ffprobe_path}'. "
              "Cannot determine video codec. Please install FFmpeg (which includes ffprobe) "
              "and ensure ffprobe is in your system PATH, or set the FFPROBE_PATH environment variable.")
    except subprocess.TimeoutExpired:
        print(f"{thread_prefix} ffprobe command timed out for '{video_path_str}'. Video might be inaccessible or too large for quick probing.")
    except subprocess.CalledProcessError as e:
        print(f"{thread_prefix} ffprobe error for '{video_path_str}'. Exit code: {e.returncode}")
        # Log stderr from ffprobe if available, as it often contains useful error details
        if e.stderr: print(f"  ffprobe stderr: {e.stderr.strip()}")
        if e.stdout: print(f"  ffprobe stdout (on error): {e.stdout.strip()}")
    except Exception as e:
        print(f"{thread_prefix} Unexpected error getting codec using ffprobe for '{video_path_str}': {e}")
        import traceback; traceback.print_exc()
    return None


class NvidiaFrameReader:
    def __init__(self, video_path, target_fps, original_fps_hint, frame_queue, stop_event,
                 dimensions_ready_event, # threading.Event instance passed from VideoProcessor
                 device_id_for_torch_output=0,
                 cuda_context_handle=0,
                 cuda_stream_handle=0):

        self.video_path = str(video_path)
        self.thread_name_prefix = f"NvidiaReader [{os.path.basename(self.video_path)}]:" # For logging

        self.target_fps = float(target_fps)
        self.original_fps_hint = float(original_fps_hint)
        self.frame_queue = frame_queue # Main output queue for (tensor, frame_num, timestamp)
        self.stop_event = stop_event # Signals this thread to stop processing
        self.dimensions_ready_event = dimensions_ready_event # Signals main thread when dimensions are known
        self.device_id_for_torch_output = int(device_id_for_torch_output) # Target CUDA device for output tensors

        # For PyNvVideoCodec.CreateDecoder if using externally managed CUDA context/stream
        self.cuda_context_handle = int(cuda_context_handle)
        self.cuda_stream_handle = int(cuda_stream_handle)

        # Attributes populated during initialization and run
        self.demuxer = None
        self.decoder = None
        self.codec_id_enum = None # PyNvVideoCodec.cudaVideoCodec enum
        self.width = 0
        self.height = 0
        self.pixel_format_from_decoder = None # PyNvVideoCodec.Pixel_Format enum
        self.actual_original_fps = 0.0
        self.frame_interval = 1.0 # Frame sampling interval
        self.next_frame_idx_to_yield = 0.0 # For frame skipping
        self.total_frames_yielded = 0 # Frames successfully put on queue

        print(f"{self.thread_name_prefix} Instance created. Target FPS: {self.target_fps:.2f}, "
              f"Orig FPS Hint: {self.original_fps_hint:.2f}")

    def _initialize_hardware_components(self):
        """
        Initializes the PyNvVideoCodec demuxer and decoder.
        This method is called from within the run() method to ensure it executes
        in the context of the reader's own thread.
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        print(f"{self.thread_name_prefix} Starting hardware component initialization...")
        try:
            # 1. Create Demuxer
            # The filename is the primary argument for CreateDemuxer.
            self.demuxer = nvc.CreateDemuxer(filename=self.video_path)
            if not self.demuxer:
                print(f"{self.thread_name_prefix} FATAL - Failed to create PyNvDemuxer for '{self.video_path}'.")
                return False
            print(f"{self.thread_name_prefix} PyNvDemuxer created successfully.")

            # 2. Determine Video Codec using ffprobe
            self.codec_id_enum = get_video_codec_nvc_enum(self.video_path, nvc) # Pass 'nvc'
            if self.codec_id_enum is None:
                print(f"{self.thread_name_prefix} CRITICAL - Failed to determine video codec via ffprobe. "
                      "Attempting to fall back to H264. This may lead to decoding failures if the "
                      "video is not H264.")
                self.codec_id_enum = nvc.cudaVideoCodec.H264 # Default fallback
            else:
                print(f"{self.thread_name_prefix} Determined video codec: {self.codec_id_enum.name}")

            # 3. Create Decoder
            # 'usedevicememory=True' is crucial for efficient GPU operation and zero-copy (e.g., with DLPack).
            # 'cudacontext' and 'cudastream' are 0 by default, meaning PyNvVideoCodec manages them internally.
            self.decoder = nvc.CreateDecoder(
                codec=self.codec_id_enum,
                cudacontext=self.cuda_context_handle,
                cudastream=self.cuda_stream_handle,
                usedevicememory=True,
                # enableasyncallocations=False # Default, not explicitly set unless a specific need arises
            )
            if not self.decoder:
                print(f"{self.thread_name_prefix} FATAL - Failed to create PyNvDecoder for codec {self.codec_id_enum.name}.")
                return False
            print(f"{self.thread_name_prefix} PyNvDecoder created (usedevicememory=True) for codec {self.codec_id_enum.name}.")
            
            print(f"{self.thread_name_prefix} Hardware component initialization successful.")
            return True

        except Exception as e:
            print(f"{self.thread_name_prefix} FATAL - Exception during hardware component initialization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _update_runtime_video_parameters(self, first_valid_decoded_frame):
        """
        Updates dynamic video parameters (width, height, FPS, pixel format)
        based on the first successfully decoded frame. Also sets the
        dimensions_ready_event to signal the main thread.
        Args:
            first_valid_decoded_frame: A non-None decoded frame object from PyNvDecoder.
        Returns:
            bool: True if parameters were successfully updated, False otherwise.
        """
        if not first_valid_decoded_frame:
            print(f"{self.thread_name_prefix} Error: _update_runtime_video_parameters called with None frame.")
            return False

        params_extracted_successfully = False
        try:
            # Attempt to get W/H from CUDA Array Interface (CAI) if DecodedFrame exposes it
            # The structure of 'decoded_frame.cuda()' and its elements is specific to PyNvVideoCodec.
            if hasattr(first_valid_decoded_frame, 'cuda') and callable(first_valid_decoded_frame.cuda):
                cai_list = first_valid_decoded_frame.cuda() # Expected to be a list of CAIMemoryView-like objects
                if cai_list and len(cai_list) > 0:
                    # Assuming the string representation is like "<CAIMemoryView [H, W, Channels_per_plane]>"
                    # This parsing is fragile; direct attributes (.shape, .height, .width) would be better.
                    shape_repr_str = str(cai_list[0]) # Use first plane for primary dimensions
                    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,', shape_repr_str)
                    if match:
                        self.height = int(match.group(1))
                        self.width = int(match.group(2))
                        params_extracted_successfully = True
                    else:
                        print(f"{self.thread_name_prefix} Warning: Could not parse H, W from CAIMemoryView string: '{shape_repr_str}'")
            
            # Fallback or alternative: Check if decoder object has GetWidth/GetHeight methods
            # These methods might only return valid values after at least one frame is decoded.
            if not params_extracted_successfully:
                if hasattr(self.decoder, 'GetWidth') and callable(self.decoder.GetWidth) and \
                   hasattr(self.decoder, 'GetHeight') and callable(self.decoder.GetHeight):
                    w, h = self.decoder.GetWidth(), self.decoder.GetHeight()
                    if w > 0 and h > 0:
                        self.width, self.height = w, h
                        params_extracted_successfully = True
                    else:
                        print(f"{self.thread_name_prefix} Warning: Decoder GetWidth/GetHeight returned non-positive values ({w}x{h}).")
            
            if not params_extracted_successfully or self.width <= 0 or self.height <= 0:
                print(f"{self.thread_name_prefix} CRITICAL - Could not determine valid frame width/height "
                      f"(current: {self.width}x{self.height}). Essential for video writer.")
                return False # Indicate failure to get crucial parameters

            # Determine Original FPS
            if self.original_fps_hint > 0:
                self.actual_original_fps = self.original_fps_hint
            else:
                # PyNvVideoCodec's decoder might not directly expose FPS.
                # If no hint, use a common default.
                self.actual_original_fps = 30.0 # Default if no hint provided
                print(f"{self.thread_name_prefix} Warning - Using default original FPS: {self.actual_original_fps:.2f} "
                      f"(original_fps_hint was {self.original_fps_hint:.2f}).")
            
            self.frame_interval = max(1.0, self.actual_original_fps / self.target_fps)
            
            # Get pixel format from the decoded frame object
            if hasattr(first_valid_decoded_frame, 'format'):
                self.pixel_format_from_decoder = first_valid_decoded_frame.format # e.g., nvc.Pixel_Format.NV12
            else:
                print(f"{self.thread_name_prefix} Warning - Could not get pixel format from decoded_frame object.")

            print(f"{self.thread_name_prefix} Video Runtime Parameters Updated Successfully: "
                  f"Dimensions: {self.width}x{self.height}, "
                  f"Original FPS: {self.actual_original_fps:.2f}, "
                  f"Target Sampling Interval: Every ~{self.frame_interval:.2f} original frames, "
                  f"Decoded Pixel Format: {self.pixel_format_from_decoder.name if self.pixel_format_from_decoder else 'Unknown'}")
            return True
        except Exception as e:
            print(f"{self.thread_name_prefix} Exception while updating runtime video parameters: {e}")
            import traceback; traceback.print_exc()
            return False # Indicate failure
        finally:
            # CRITICAL: Always set the dimensions_ready_event, even if parameter update failed.
            # This prevents the main thread from waiting indefinitely on self.dimensions_ready_event.wait().
            # The main thread should check self.width and self.height after the event is set.
            if self.dimensions_ready_event and not self.dimensions_ready_event.is_set():
                print(f"{self.thread_name_prefix} Setting dimensions_ready_event (from _update_runtime_video_parameters).")
                self.dimensions_ready_event.set()

    def run(self):
        print(f"{self.thread_name_prefix} Run method started.")
        
        # Step 1: Initialize hardware components (demuxer, decoder)
        initialization_successful = self._initialize_hardware_components()
        
        if not initialization_successful:
            print(f"{self.thread_name_prefix} Hardware initialization failed. Thread will exit.")
            # Ensure dimensions_ready_event is set to unblock the main thread if it's waiting.
            if self.dimensions_ready_event and not self.dimensions_ready_event.is_set():
                print(f"{self.thread_name_prefix} Setting dimensions_ready_event (due to init failure) to unblock main thread.")
                self.dimensions_ready_event.set()
            # Signal consumer (e.g., VideoProcessor's main loop) that no frames will come.
            try: self.frame_queue.put(None, block=False, timeout=0.2)
            except queue.Full: print(f"{self.thread_name_prefix} Frame queue full while sending EOS on init failure.")
            except Exception as e: print(f"{self.thread_name_prefix} Error sending EOS on init failure: {e}")
            return # Exit thread

        # Initialize counters and flags for the processing loop
        raw_decoded_frame_count = 0       # Total frames output by the decoder
        params_updated_from_first_frame_flag = False # True after first frame params are read
        demuxer_packet_count_processed = 0     # Packets processed from demuxer

        try:
            print(f"{self.thread_name_prefix} Starting main demuxing and decoding loop.")
            for packet_data in self.demuxer: # Iterates through compressed packets
                if self.stop_event.is_set():
                    print(f"{self.thread_name_prefix} Stop event detected during demuxing. Breaking loop.")
                    break
                
                demuxer_packet_count_processed += 1
                if not packet_data or packet_data.bsl == 0: # bsl is bitstream length
                    # This can happen (e.g. empty packets), usually not an error.
                    # print(f"{self.thread_name_prefix} Received empty or invalid packet data (packet {demuxer_packet_count_processed}). Skipping.")
                    continue

                try:
                    # PyNvVideoCodec.PyNvDecoder.Decode() returns an iterable (often a generator) of decoded frames.
                    # A single packet can sometimes produce multiple frames (e.g. some codecs, or flush).
                    decoded_frames_iterable = self.decoder.Decode(packet_data)
                    if decoded_frames_iterable is None: # Should not happen if API is well-behaved
                        # print(f"{self.thread_name_prefix} decoder.Decode() returned None for packet {demuxer_packet_count_processed}. Skipping packet.")
                        continue
                except Exception as dec_err:
                    print(f"{self.thread_name_prefix} Exception calling decoder.Decode() for packet {demuxer_packet_count_processed}: {dec_err}")
                    # Depending on severity, might continue or break. For now, continue.
                    continue 

                frames_decoded_from_this_packet = 0
                for decoded_frame_object in decoded_frames_iterable:
                    if self.stop_event.is_set():
                        print(f"{self.thread_name_prefix} Stop event detected during inner frame decoding loop.")
                        break 
                    
                    if decoded_frame_object is None: continue # Should not happen from a well-behaved iterable

                    frames_decoded_from_this_packet += 1
                    raw_decoded_frame_count += 1

                    # Update runtime parameters (W, H, FPS) using the very first valid decoded frame
                    if not params_updated_from_first_frame_flag:
                        update_success = self._update_runtime_video_parameters(decoded_frame_object)
                        params_updated_from_first_frame_flag = True # Mark as attempted
                        if not update_success or self.width <= 0 or self.height <= 0:
                            print(f"{self.thread_name_prefix} CRITICAL - Failed to get valid video parameters "
                                  f"(W:{self.width},H:{self.height}) from the first decoded frame. Stopping thread.")
                            self.stop_event.set() # Signal critical failure
                            break # Break from inner loop

                    # Frame skipping logic (can only be applied after params are known)
                    if raw_decoded_frame_count < self.next_frame_idx_to_yield:
                        continue

                    # Convert decoded frame to PyTorch tensor using DLPack (zero-copy if usedevicememory=True)
                    try:
                        gpu_tensor = torch.from_dlpack(decoded_frame_object)
                        # Optional: Ensure tensor is on the target GPU if multiple GPUs are involved.
                        # DLPack should typically create the tensor on the same device as the source data.
                        # if gpu_tensor.device.index != self.device_id_for_torch_output:
                        #     gpu_tensor = gpu_tensor.to(torch.device(f'cuda:{self.device_id_for_torch_output}'))
                    except Exception as e_dlpack:
                        print(f"{self.thread_name_prefix} ERROR - Failed to convert decoded_frame (raw frame num "
                              f"~{raw_decoded_frame_count}) to PyTorch tensor via DLPack: {e_dlpack}")
                        # Decide how to handle: skip this frame or stop? For now, skip.
                        if params_updated_from_first_frame_flag: self.next_frame_idx_to_yield += self.frame_interval
                        continue # Skip to next decoded frame or packet
                    
                    # Calculate timestamp (seconds from the start of the video)
                    current_frame_timestamp_sec = (raw_decoded_frame_count - 1) / self.actual_original_fps if self.actual_original_fps > 0 else 0.0
                    
                    # Prepare item for the queue
                    item_to_queue = (gpu_tensor, raw_decoded_frame_count, current_frame_timestamp_sec)
                    
                    # Put item on the frame_queue with timeout and retry logic
                    put_successful = False
                    for _attempt in range(2): # Try putting twice if queue is full
                        if self.stop_event.is_set(): break # Check before attempting put
                        try:
                            self.frame_queue.put(item_to_queue, block=True, timeout=0.5) # Shorter timeout for first try
                            self.total_frames_yielded += 1
                            if params_updated_from_first_frame_flag: self.next_frame_idx_to_yield += self.frame_interval
                            put_successful = True
                            break # Successfully put
                        except queue.Full:
                            if _attempt == 0: # First attempt failed
                                print(f"{self.thread_name_prefix} Frame queue full, waiting briefly...")
                                if self.stop_event.wait(timeout=0.2): # Wait a bit, check stop_event
                                    print(f"{self.thread_name_prefix} Stop event detected while queue was full.")
                                    break # Break from retry loop
                            else: # Second attempt also failed
                                print(f"{self.thread_name_prefix} CRITICAL - Frame queue persistently full. Stopping thread.")
                                self.stop_event.set() # Signal critical failure to stop everything
                        except Exception as q_err:
                             print(f"{self.thread_name_prefix} CRITICAL - Error putting frame on queue: {q_err}")
                             self.stop_event.set(); break
                    if not put_successful or self.stop_event.is_set(): break # Break inner loop if not put or stopped
                
                if self.stop_event.is_set(): break # Break outer packet loop if stop was set in inner loop
                # Optional: Log if a non-empty packet yielded zero displayable frames
                # if frames_decoded_from_this_packet == 0 and packet_data.bsl > 0:
                #    print(f"{self.thread_name_prefix} Decoder yielded 0 frames for non-empty packet {demuxer_packet_count_processed} (size: {packet_data.bsl}).")

            if not self.stop_event.is_set():
                 print(f"{self.thread_name_prefix} Demuxer finished iterating all {demuxer_packet_count_processed} packets (End Of Stream).")
            
            if demuxer_packet_count_processed == 0 and not self.stop_event.is_set():
                print(f"{self.thread_name_prefix} WARNING - Demuxer yielded 0 packets. Video file might be empty, corrupted, or demuxer issue.")

        except Exception as main_loop_err:
            print(f"{self.thread_name_prefix} CRITICAL - Unhandled exception in main decoding loop: {main_loop_err}")
            import traceback; traceback.print_exc()
            self.stop_event.set() # Signal overall failure
        finally:
            # Ensure dimensions_ready_event is set, especially if loop exited early,
            # or if params were never updated (e.g. demuxer gave 0 packets).
            if self.dimensions_ready_event and not self.dimensions_ready_event.is_set():
                print(f"{self.thread_name_prefix} Setting dimensions_ready_event in finally block (to ensure main thread unblocks).")
                self.dimensions_ready_event.set()

            final_stats_msg = (f"Raw frames decoded by hardware: {raw_decoded_frame_count}, "
                               f"Frames yielded to processing queue: {self.total_frames_yielded}.")
            print(f"{self.thread_name_prefix} Exiting run method. {final_stats_msg}")
            
            # Signal end to consumer by putting None in the queue (important for VideoProcessor's loop)
            try:
                self.frame_queue.put(None, block=False, timeout=1.0) # Use non-blocking or short timeout
                print(f"{self.thread_name_prefix} EOS sentinel (None) successfully placed on frame_queue.")
            except queue.Full:
                print(f"{self.thread_name_prefix} Warning - Frame queue was full while trying to send EOS sentinel. Consumer might miss it or be blocked.")
            except Exception as e_eos:
                print(f"{self.thread_name_prefix} Error sending EOS sentinel to frame_queue: {e_eos}")
            
            # PyNvVideoCodec objects (demuxer, decoder) are C++ RAII-backed.
            # They should release GPU resources when their Python objects are garbage collected.
            # Explicit deletion (del self.decoder; del self.demuxer) is usually not strictly necessary
            # but can be added if resource leakage is suspected.
            print(f"{self.thread_name_prefix} GPU resources for demuxer/decoder should be released by Python's Garbage Collector.")