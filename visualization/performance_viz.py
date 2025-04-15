# project/visualization/performance_viz.py
import os
import numpy as np
from config import ENABLE_DETAILED_PERFORMANCE_METRICS

# Optional import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

def visualize_performance(perf_tracker):
    if not ENABLE_DETAILED_PERFORMANCE_METRICS: print("Detailed metrics off, skipping viz."); return
    if not MATPLOTLIB_AVAILABLE: print("Matplotlib not found, skipping viz."); return
    if perf_tracker is None or not hasattr(perf_tracker, 'get_summary'): print("Perf tracker invalid."); return

    print("Generating performance visualizations..."); output_dir = "performance_charts"; os.makedirs(output_dir, exist_ok=True)
    try:
        summary = perf_tracker.get_summary()

        # Chart 1: Timing Breakdown
        fig1, ax1 = plt.subplots(figsize=(7, 7)); labels, sizes, other_time = [], [], 0; threshold = 0.01
        total_time = summary['timings'].get('total', 1); total_time = max(total_time, 1e-6)
        sorted_timings = sorted(summary['timings'].items(), key=lambda item: item[1], reverse=True)
        for key, value in sorted_timings:
            if key != 'total':
                if value / total_time >= threshold: labels.append(key.replace('_',' ').title()); sizes.append(value)
                elif value > 0: other_time += value
        if other_time > 0: labels.append("Other"); sizes.append(other_time)
        if sizes:
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85); ax1.axis('equal'); ax1.set_title('Processing Time Breakdown')
            fig1.gca().add_artist(plt.Circle((0,0),0.70,fc='white')); plt.tight_layout(); fig1.savefig(os.path.join(output_dir, "timing_breakdown.png"), dpi=150); plt.close(fig1)
        else: print("No timing data for pie chart.")

        # Chart 2: GPU/CPU Utilization
        gpu_util = perf_tracker.gpu_metrics.get('utilization', []); cpu_util = perf_tracker.cpu_metrics.get('utilization', [])
        if gpu_util or cpu_util:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            if gpu_util: ax2.plot(gpu_util, label='GPU Util (%)', color='r')
            if cpu_util: ax2.plot(cpu_util, label='CPU Util (%)', color='b')
            ax2.set_title('Resource Utilization'); ax2.set_xlabel('Sample Point'); ax2.set_ylabel('Utilization (%)'); ax2.legend(); ax2.grid(True, alpha=0.3); plt.tight_layout(); fig2.savefig(os.path.join(output_dir, "resource_utilization.png"), dpi=150); plt.close(fig2)
        else: print("No CPU/GPU utilization data.")

        # Chart 3: FPS Over Time
        b_sizes = perf_tracker.batch_metrics.get('batch_sizes', []); b_times = perf_tracker.batch_metrics.get('batch_times', [])
        if b_sizes and b_times and len(b_sizes) > 1:
            b_fps = [(s/t if t>0 else 0) for s,t in zip(b_sizes,b_times)]; fig3, ax3 = plt.subplots(figsize=(10,5))
            ax3.plot(b_fps, label='Batch FPS', alpha=0.7)
            if len(b_fps) > 10:
                window = min(30, len(b_fps)//5); mov_avg = np.convolve(b_fps, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(b_fps)), mov_avg, 'r-', label=f'{window}-Batch Moving Avg FPS')
            ax3.set_title('Processing Speed'); ax3.set_xlabel('Batch'); ax3.set_ylabel('FPS'); ax3.legend(); ax3.grid(True, alpha=0.3); plt.tight_layout(); fig3.savefig(os.path.join(output_dir, "fps_over_time.png"), dpi=150); plt.close(fig3)
        else: print("Not enough batch data for FPS chart.")
        print(f"Performance charts saved to '{output_dir}'.")
    except Exception as e: print(f"Error generating viz: {e}")
