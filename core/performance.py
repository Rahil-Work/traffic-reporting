# project/core/performance.py
import time
import torch
import os
import gc # Make sure gc is imported
import numpy as np # Import numpy
from datetime import timedelta
from config import (
    ENABLE_DETAILED_PERFORMANCE_METRICS, PERFORMANCE_SAMPLE_INTERVAL,
    DEBUG_MODE
)

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    if DEBUG_MODE: print("psutil not found, CPU/RAM monitoring disabled.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
    if DEBUG_MODE: print("pynvml initialized for GPU monitoring.")
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None
    if DEBUG_MODE: print("pynvml not found, detailed GPU monitoring disabled.")
except pynvml.NVMLError as e:
    PYNVML_AVAILABLE = False
    pynvml = None
    if DEBUG_MODE: print(f"Could not initialize pynvml: {e}")


class PerformanceTracker:
    """Comprehensive performance tracking system."""

    def __init__(self, enabled=ENABLE_DETAILED_PERFORMANCE_METRICS):
        self.enabled = enabled
        # Basic metrics
        self.frames_processed = 0
        self.frames_total = 0
        self.processing_start_time = None
        self.processing_end_time = None
        self.total_time = 0.0
        self.detections_count = 0

        if not self.enabled:
            self.timings = {'total': 0.0}
            return

        # Detailed Timers
        self.timings = {
            'video_read': 0.0, 'preprocessing': 0.0, 'inference': 0.0,
            'tracking_update': 0.0, 'zone_checking': 0.0, # Split tracking time
            'drawing': 0.0, 'video_write': 0.0, 'memory_cleanup': 0.0,
            'overhead': 0.0, 'total': 0.0
        }
        self._timer_starts = {}

        # GPU Metrics
        self.gpu_metrics = {'memory_allocated': [], 'memory_reserved': [], 'utilization': []}
        self._gpu_handle = None
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                if DEBUG_MODE: print(f"Failed to get NVML handle: {e}")
                self._gpu_handle = None

        # CPU Metrics
        self.cpu_metrics = {'utilization': [], 'memory_percent': []}
        self.system_metrics = {'ram_usage_gb': []}

        # Vehicle Tracking Metrics
        self.vehicle_metrics = {
            'entries': 0, 'exits_valid': 0, 'exits_timeout': 0,
            'exits_forced': 0, 'tracking_errors': 0,
            'total_time_in_intersection': 0.0,
            'avg_time_in_intersection': 0.0, 'tracking_accuracy': 0.0
        }

        # Batch Processing Metrics
        self.batch_metrics = {
            'batch_sizes': [], 'batch_times': [],
            'avg_batch_size': 0, 'avg_batch_time': 0.0, 'batch_throughput': 0.0
        }

        # Sampling state
        self.last_sample_time = time.monotonic()
        self.sample_interval = PERFORMANCE_SAMPLE_INTERVAL

        # CUDA Events
        self.cuda_events_enabled = torch.cuda.is_available()
        if self.cuda_events_enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = self.end_event = None

    # --- Core Timing ---
    def start_processing(self, total_frames=0):
        self.processing_start_time = time.monotonic()
        self.frames_total = total_frames
        if self.enabled:
            self.last_sample_time = self.processing_start_time

    def end_processing(self):
        if self.processing_start_time is None: return
        self.processing_end_time = time.monotonic()
        self.total_time = self.processing_end_time - self.processing_start_time
        self.timings['total'] = self.total_time
        if self.enabled:
            self._calculate_derived_metrics()

    def start_timer(self, key):
        if not self.enabled: return
        self._timer_starts[key] = time.monotonic()

    def end_timer(self, key):
        if not self.enabled or key not in self._timer_starts: return 0
        end_time = time.monotonic()
        duration = end_time - self._timer_starts.pop(key)
        self.timings[key] = self.timings.get(key, 0.0) + duration
        return duration

    def record_inference_time_gpu(self, start_event, end_event):
        if not self.enabled or not self.cuda_events_enabled: return
        try:
            # Assumes events already recorded and synchronized
            duration_ms = start_event.elapsed_time(end_event)
            self.timings['inference'] += duration_ms / 1000.0
        except Exception as e:
             if DEBUG_MODE: print(f"Error recording GPU inference time: {e}")

    # --- Metric Recording ---
    def record_batch_processed(self, batch_size, batch_time):
        self.frames_processed += batch_size
        if self.enabled and batch_size > 0 and batch_time > 0:
            self.batch_metrics['batch_sizes'].append(batch_size)
            self.batch_metrics['batch_times'].append(batch_time)

    def record_detection(self, count=1):
        self.detections_count += count

    def record_vehicle_entry(self):
        if self.enabled: self.vehicle_metrics['entries'] += 1

    def record_vehicle_exit(self, exit_status, time_in_intersection=None):
        if not self.enabled: return
        total_valid_exits = self.vehicle_metrics['exits_valid'] # Get current count before incrementing

        if exit_status == 'exited':
            self.vehicle_metrics['exits_valid'] += 1
            if time_in_intersection is not None:
                 self.vehicle_metrics['total_time_in_intersection'] += time_in_intersection
            total_valid_exits += 1 # Update for average calculation
        elif exit_status == 'timed_out':
            self.vehicle_metrics['exits_timeout'] += 1
        elif exit_status == 'forced_exit':
             self.vehicle_metrics['exits_forced'] += 1

        # Update average time using the *new* count of valid exits
        if exit_status == 'exited' and total_valid_exits > 0 and time_in_intersection is not None:
            self.vehicle_metrics['avg_time_in_intersection'] = \
                self.vehicle_metrics['total_time_in_intersection'] / total_valid_exits

    def record_tracking_error(self, count=1):
         if self.enabled: self.vehicle_metrics['tracking_errors'] += count

    # --- System Sampling ---
    def sample_system_metrics(self, force=False):
        if not self.enabled: return
        current_time = time.monotonic()
        if not force and (current_time - self.last_sample_time < self.sample_interval):
            return

        # GPU Metrics
        if torch.cuda.is_available():
            try:
                 alloc = torch.cuda.memory_allocated() / 1e9
                 res = torch.cuda.memory_reserved() / 1e9
                 self.gpu_metrics['memory_allocated'].append(alloc)
                 self.gpu_metrics['memory_reserved'].append(res)
                 gpu_util = 0
                 if self._gpu_handle:
                     util_rates = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                     gpu_util = util_rates.gpu
                 self.gpu_metrics['utilization'].append(gpu_util)
            except Exception as e:
                 if DEBUG_MODE: print(f"Error sampling GPU metrics: {e}")
                 # Append 0 to maintain list lengths if error
                 self.gpu_metrics['memory_allocated'].append(0)
                 self.gpu_metrics['memory_reserved'].append(0)
                 self.gpu_metrics['utilization'].append(0)

        # CPU / RAM Metrics
        if PSUTIL_AVAILABLE:
            try:
                 self.cpu_metrics['utilization'].append(psutil.cpu_percent(interval=None))
                 mem_info = psutil.virtual_memory()
                 self.cpu_metrics['memory_percent'].append(mem_info.percent)
                 self.system_metrics['ram_usage_gb'].append(mem_info.used / 1e9)
            except Exception as e:
                if DEBUG_MODE: print(f"Error sampling CPU/RAM metrics: {e}")
                self.cpu_metrics['utilization'].append(0)
                self.cpu_metrics['memory_percent'].append(0)
                self.system_metrics['ram_usage_gb'].append(0)

        self.last_sample_time = current_time

    # --- Reporting & Calculation ---
    def _calculate_derived_metrics(self):
        if not self.enabled or self.frames_processed == 0: return

        # Batch metrics
        if self.batch_metrics['batch_times']:
            self.batch_metrics['avg_batch_size'] = np.mean(self.batch_metrics['batch_sizes']) if self.batch_metrics['batch_sizes'] else 0
            self.batch_metrics['avg_batch_time'] = np.mean(self.batch_metrics['batch_times']) if self.batch_metrics['batch_times'] else 0
            if self.batch_metrics['avg_batch_time'] > 0:
                 self.batch_metrics['batch_throughput'] = self.batch_metrics['avg_batch_size'] / self.batch_metrics['avg_batch_time']
            else: self.batch_metrics['batch_throughput'] = 0

        # Vehicle metrics
        entries = self.vehicle_metrics['entries']
        valid_exits = self.vehicle_metrics['exits_valid']
        self.vehicle_metrics['tracking_accuracy'] = (valid_exits / entries) if entries > 0 else 0.0

        # Overhead time
        accounted_time = sum(v for k, v in self.timings.items() if k not in ['total', 'overhead'])
        self.timings['overhead'] = max(0, self.total_time - accounted_time)

    def get_progress(self):
        if self.processing_start_time is None or self.frames_processed == 0:
            return {'percent': 0, 'fps': 0, 'eta': timedelta(seconds=0), 'elapsed': timedelta(seconds=0)}

        elapsed_seconds = time.monotonic() - self.processing_start_time
        fps = self.frames_processed / elapsed_seconds if elapsed_seconds > 0 else 0

        if self.frames_total > 0:
            percent = min(100.0, (self.frames_processed / self.frames_total) * 100)
            eta_seconds = (self.frames_total - self.frames_processed) / fps if fps > 0 else 0
        else: percent, eta_seconds = 0, 0

        return {
            'percent': percent, 'fps': fps,
            'eta': timedelta(seconds=int(eta_seconds)),
            'elapsed': timedelta(seconds=int(elapsed_seconds))
        }

    def get_summary(self):
        if self.processing_end_time is None: self.end_processing()

        basic_summary = {
            'processing': {
                'frames_processed': self.frames_processed, 'frames_total': self.frames_total,
                'detections_count': self.detections_count, 'total_time_seconds': self.total_time,
                'overall_fps': self.frames_processed / self.total_time if self.total_time > 0 else 0
            }}
        if not self.enabled: return basic_summary

        avg_metrics = {
             'gpu_mem_allocated_gb': np.mean(self.gpu_metrics['memory_allocated']) if self.gpu_metrics['memory_allocated'] else 0,
             'gpu_mem_reserved_gb': np.mean(self.gpu_metrics['memory_reserved']) if self.gpu_metrics['memory_reserved'] else 0,
             'gpu_utilization_percent': np.mean(self.gpu_metrics['utilization']) if self.gpu_metrics['utilization'] else 0,
             'cpu_utilization_percent': np.mean(self.cpu_metrics['utilization']) if self.cpu_metrics['utilization'] else 0,
             'cpu_memory_percent': np.mean(self.cpu_metrics['memory_percent']) if self.cpu_metrics['memory_percent'] else 0,
             'ram_usage_gb': np.mean(self.system_metrics['ram_usage_gb']) if self.system_metrics['ram_usage_gb'] else 0,
        }
        timing_percent = {k: (v / self.total_time * 100 if self.total_time > 0 else 0) for k, v in self.timings.items()}

        return { **basic_summary, 'timings': self.timings, 'timing_percent': timing_percent,
                 'avg_system_metrics': avg_metrics, 'vehicle_metrics': self.vehicle_metrics,
                 'batch_metrics': self.batch_metrics }

    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "="*80 + "\n" + " PERFORMANCE SUMMARY ".center(80, "=") + "\n" + "="*80)
        proc = summary['processing']
        print(f"Frames Processed: {proc['frames_processed']:,} / {proc.get('frames_total', 'N/A'):,}")
        print(f"Total Time:       {proc['total_time_seconds']:.2f} seconds ({timedelta(seconds=int(proc['total_time_seconds']))})")
        print(f"Overall Speed:    {proc['overall_fps']:.2f} FPS")
        print(f"Total Detections: {proc['detections_count']:,}")

        if not self.enabled:
            print("\nDetailed performance metrics disabled." + "\n" + "="*80); return

        print("\n--- Timing Breakdown ---")
        sorted_timings = sorted(summary['timings'].items(), key=lambda item: item[1], reverse=True)
        for key, value in sorted_timings:
             if key != 'total' and value > 1e-4: # Print non-zero timings
                 percent = summary['timing_percent'].get(key, 0)
                 print(f"  {key.replace('_', ' ').title():<20}: {value:>8.3f}s ({percent:5.1f}%)")

        print("\n--- Average System Utilization ---")
        sysm = summary['avg_system_metrics']
        if torch.cuda.is_available():
             print(f"  GPU Utilization:    {sysm['gpu_utilization_percent']:>6.1f}%")
             print(f"  GPU Memory (Alloc): {sysm['gpu_mem_allocated_gb']:>6.2f} GB")
             print(f"  GPU Memory (Reserv):{sysm['gpu_mem_reserved_gb']:>6.2f} GB")
        print(f"  CPU Utilization:    {sysm['cpu_utilization_percent']:>6.1f}%")
        print(f"  CPU Memory Usage:   {sysm['cpu_memory_percent']:>6.1f}%")
        print(f"  System RAM Usage:   {sysm['ram_usage_gb']:>6.2f} GB")

        print("\n--- Vehicle Tracking ---")
        vm = summary['vehicle_metrics']
        print(f"  Entries:            {vm['entries']:,}")
        print(f"  Valid Exits:        {vm['exits_valid']:,}")
        print(f"  Timeouts:           {vm['exits_timeout']:,}")
        print(f"  Forced Exits:       {vm['exits_forced']:,}")
        print(f"  Tracking Accuracy:  {vm['tracking_accuracy'] * 100:.1f}%")
        print(f"  Avg. Time in Inter: {vm['avg_time_in_intersection']:.2f}s (Valid Exits)")

        print("\n--- Batch Processing ---")
        bm = summary['batch_metrics']
        print(f"  Average Batch Size: {bm['avg_batch_size']:.1f}")
        print(f"  Average Batch Time: {bm['avg_batch_time']:.4f}s")
        print(f"  Batch Throughput:   {bm['batch_throughput']:.2f} FPS")
        print("="*80)

    def __del__(self):
         if PYNVML_AVAILABLE and hasattr(pynvml, 'nvmlShutdown'):
             try: pynvml.nvmlShutdown()
             except: pass # Ignore shutdown errors