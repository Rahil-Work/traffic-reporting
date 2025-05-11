import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import comet_ml
import torch
from ultralytics import YOLO
import psutil
import gc

# =========================================
# Configurations
# =========================================
COMET_PROJECT_NAME = "vehicle-detection"

DATA_YAML = r"/home/merged_10052025/dataset_split/data.yaml"


# Models to train with specific batch sizes for RTX 3090 (24GB)
# Batch sizes are estimates; monitor VRAM and adjust if needed.
MODEL_CONFIGS = {
    "yolov8m": {
        "pt_file": "yolov8m.pt",
        "batch_size": 96, 
        "note": "YOLOv8 Medium. Target: 48-64. Reduce if OOM."
    },
    "rtdetr-l": {
        "pt_file": "rtdetr-l.pt",
        "batch_size": 16,  # Increased from 12. (RT-DETR can be heavier)
        "note": "RT-DETR Large. Target: 20-32. Reduce if OOM."
    },
    "yolov8l": {
        "pt_file": "yolov8l.pt",
        "batch_size": 48,  # Increased from 16. (If BS16~6.5GB, BS40~16GB. Could try 48)
        "note": "YOLOv8 Large. Target: 32-48. Reduce if OOM."
    },
}

IMG_SIZE = 416       # Standard size, well-suited for RTX 3090 performance.
EPOCHS = 250         # Number of training epochs
PATIENCE = 40       # Increased patience for early stopping
# OPTIMIZER = 'SGD'    # Fixed optimizer as requested
# Given 125Gi RAM and 36 physical / 72 logical cores:
# Option 1: Start by matching physical cores
WORKERS = 36
num_cpus = psutil.cpu_count(logical=True)
# Option 2: Or try a more aggressive value if 36 works well
# WORKERS = 48
# WORKERS = 56
# WORKERS = 60
# WORKERS = 64
# if num_cpus:
#     WORKERS = min(WORKERS, num_cpus) # Cap at available logical cores
# else:
#     WORKERS = 8 # Default if psutil fails for some reason

# # Ensure WORKERS does not exceed a reasonable cap if num_cpus is unusually high for some reason
# # For your 28 cores, directly setting to 14, 20, 24, or 28 is fine.
# if num_cpus: # Only if num_cpus is not None or 0
#     WORKERS = min(WORKERS, num_cpus) # Don't set more workers than available logical cores
# else: # Fallback if psutil fails for some reason
#     WORKERS = 8

print("CPU count:",num_cpus)
print(f"INFO: Using {WORKERS} DataLoader workers.")

# Parameter grid for SGD
PARAMETER_GRID_ADVANCED = [
    # --- Baseline Set (similar to your original 'SGD_Base') ---
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937,
    #  'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, # Base HSV
    #  'degrees': 10.0, 'translate': 0.1, 'scale': 0.6, 'shear': 2.0, 'perspective': 0.001,
    #  'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.15, 'copy_paste': 0.0, # No copy_paste initially
    #  'cls': 1.5, 'box': 7.5, 'dfl': 1.5, # Default loss gains
    #  'label_smoothing': 0.05,
    #  'brightness': 0.0, 'contrast': 0.0, # No explicit brightness/contrast initially
    #  'multi_scale': False, # Multi-scale off for baseline
    #  'run_tag': 'SGD_Adv_Baseline'},

    # --- Explore Learning Rate & Weight Decay ---
    # {'lr0': 0.008, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.15, 'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_LR008'},
    # {'lr0': 0.01,  'weight_decay': 0.00075,'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.15, 'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_WD00075'},

    # --- HSV Augmentation Variations ---
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.9, 'hsv_v': 0.6, 'mixup': 0.15, 'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_HSVe'}, # Extreme HSV
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.5, 'hsv_v': 0.3, 'mixup': 0.15, 'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_HSVl'}, # Lower HSV

    # --- Brightness/Contrast Augmentation ---
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.15, 'cls': 1.25, 'label_smoothing': 0.05,
     # 'brightness': 0.15, 'contrast': 0.15, 'run_tag': 'SGD_Adv_BrightContrast015'},
    
    # --- Mixup & Label Smoothing Variations ---
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.25, 'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_Mixup025'},
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.0,  'cls': 1.5, 'label_smoothing': 0.05, 'run_tag': 'SGD_Adv_NoMixup'},
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.15, 'label_smoothing': 0.1, 'run_tag': 'SGD_Adv_LS010'},

    # All auto
    {'run_tag': 'all_auto'},

    # --- Loss Gain Variations ---
    {'lr0': 0.01,'cls': 0.5, 'box': 10.0, 'dfl': 1.5, 'optimizer':'SGD', 'run_tag': 'SGD_Adv_HighBoxLowCls'},
    {'lr0': 0.01,'cls': 2., 'box': 5.0, 'dfl': 1.5, 'optimizer':'SGD', 'run_tag': 'SGD_Adv_HighClsHighBox'},
    
    # --- Experiment with Mosaic (e.g., turning it off or longer close_mosaic) ---
    # {'lr0': 0.01, 'weight_decay': 0.0005, 'momentum': 0.937, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mixup': 0.15, 'cls': 1.5, 'label_smoothing': 0.05,
    #  'mosaic': 0.0, 'run_tag': 'SGD_Adv_NoMosaic'}, # Turning mosaic off entirely


    # --- A more "Aggressive" Augmentation Set (combining a few positives) ---
    {'lr0': 0.001, 'weight_decay': 0.0005, 'momentum': 0.937,
     'hsv_h': 0.020, 'hsv_s': 0.8, 'hsv_v': 0.5, # Slightly more HSV
     'degrees': 15.0, 'translate': 0.15, 'scale': 0.7, 'shear': 3.0, 'perspective': 0.0015, # More geometric
     'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.20, 'copy_paste': 0.05, # Moderate mixup, slight copy_paste
     'cls': 1.25, 'box': 7.5, 'dfl': 1.5,
     'label_smoothing': 0.075,
     'brightness': 0.1, 'contrast': 0.1, # Add brightness/contrast
     'run_tag': 'SGD_Adv_AggressiveAug_LowLR'},
]

PARAMETER_GRID_ADVANCED = [
    {
        "run_tag": "balanced_tracking_focus",
        "optimizer": "AdamW",
        "lr0": 0.03,              # Increased LR to account for larger batch size
        "lrf": 0.01,              # Final LR fraction
        "weight_decay": 0.0005,
        "warmup_epochs": 2.0,     # Shorter warmup with larger batches
        "box": 8.5,               # Increased for better tracking precision
        "cls": 1.2,               # Increased for better classification with unbalanced classes
        "dfl": 1.0,               # Slightly reduced
        "cos_lr": True,
        "pose":0,
        "kobj":0,
    },
    
    {
        "run_tag": "classification_priority",
        "optimizer": "AdamW",
        "lr0": 0.025,             # Slightly lower than baseline but still elevated for batch size
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0007,   # Increased to help with class imbalance
        "warmup_epochs": 2.0,
        "box": 7.5,               # Standard box weight
        "cls": 2.0,               # Significantly increased for classification focus
        "dfl": 1.2,
        "cos_lr": True,
        "pose":0,
        "kobj":0,
    },
    
    {
        "run_tag": "tracking_precision",
        "optimizer": "SGD",       # Testing SGD for potentially better localization
        "lr0": 0.04,              # Higher LR for SGD with large batch size
        "lrf": 0.01,
        "momentum": 0.95,         # Increased momentum for tracking stability
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,     # Longer warmup for SGD stability
        "box": 10.0,              # Very high box weight for tracking precision
        "cls": 1.0,               # Moderate classification weight
        "dfl": 1.5,               # Standard DFL weight
        "cos_lr": True,
        "pose":0,
        "kobj":0,
    },
    
    {
        "run_tag": "balanced_detection",
        "optimizer": "AdamW",
        "lr0": 0.03,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 2.0,
        "box": 8.0,               # Moderately increased box weight
        "cls": 1.5,               # Increased classification weight
        "dfl": 0.8,               # Slightly reduced DFL
        "cos_lr": True,
        "pose":0,
        "kobj":0,
    },
    
    {
        "run_tag": "extreme_localization",
        "optimizer": "AdamW,
        "lr0": 0.02,              # Lower than baseline but still adjusted for batch size
        "lrf": 0.007,             # Lower final learning rate
        "momentum": 0.94,
        "weight_decay": 0.0003,   # Lower weight decay for extreme specialization
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.85,
        "box": 12.0,              # Extremely high box weight
        "cls": 0.8,               # Moderate classification weight
        "dfl": 2.0,               # Increased DFL for precise boundaries
        "cos_lr": True,
        "pose":0,
        "kobj":0,
    },

    {'run_tag': 'all_auto'},    
]

# =========================================
# Training Function
# =========================================
def run_training(model_name_key, model_pt_file, model_batch_size, run_index_overall, params_from_grid):
    # All hyperparams, including augmentations, are now expected to be in params_from_grid
    current_run_params = params_from_grid.copy() # Work with a copy

    param_run_tag = current_run_params.pop("run_tag", f"ParamSet{run_index_overall+1}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Construct a detailed run name
    # lr_str = f"LR{current_run_params.get('lr0', -1):.4f}".replace('.', 'p')
    # wd_str = f"WD{current_run_params.get('weight_decay', 'def'):.5f}".replace('.', 'p')
    run_name = f"{param_run_tag}_{timestamp}"

    print(f"\n--- Starting Run: {run_name} ---")
    print(f"Model: {model_pt_file}, Batch Size: {model_batch_size}")
    print(f"Full Params for this run: {current_run_params}")

    # Base training args that are constant for all runs in this script
    training_args = {
        'data': str(DATA_YAML),
        'model': str(model_pt_file),
        'imgsz': IMG_SIZE,
        'batch': model_batch_size,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        # 'optimizer': OPTIMIZER,
        'save': True,
        # 'save_period': SAVE_PERIOD,
        'project': COMET_PROJECT_NAME,
        'name': run_name,
        'exist_ok': False,
        'verbose': True,
        'workers': WORKERS,
        'nbs': model_batch_size, # Match nominal batch size to actual batch size
        'plots': True
    }

    # Update training_args with all parameters from the current set in the grid
    # This includes lr0, weight_decay, hsv_s, degrees, mixup, etc.
    training_args.update(current_run_params)

    yolo_model = None
    try:
        if "COMET_API_KEY" not in os.environ:
            os.environ["COMET_API_KEY"] = "sT4oj3U5RYVj2evCBfBG2UVwa" # Replace with your key or set as env var
            print("INFO: Using default Comet API Key.")

        if not os.environ.get("COMET_API_KEY"):
            print("WARNING: COMET_API_KEY not set. Comet.ml logging will be disabled.")

        yolo_model = YOLO(training_args['model'])

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            training_args['device'] = '0' if num_gpus == 1 else ','.join(str(i) for i in range(num_gpus))
            print(f"INFO: Using CUDA device(s): {training_args['device']}")
        else:
            training_args['device'] = 'cpu'
            print("WARNING: Running on CPU. This will be significantly slower.")

        yolo_model.train(**training_args)
        print(f"--- [Run Successful] Training complete for {run_name} ---")

    except Exception as e:
        print(f"--- [Run ERROR] for {run_name}: {e} ---")
        traceback.print_exc()
    finally:
        if yolo_model: # Check if yolo_model was initialized
            del yolo_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        print(f"--- Finished processing for {run_name} ---")
        time.sleep(5)

# =========================================
# Main Execution
# =========================================
if __name__ == '__main__':
    if not Path(DATA_YAML).exists():
        print(f"ERROR: DATA_YAML path not found: {DATA_YAML}")
        print("Please verify the DATA_YAML path in the script.")
        exit(1)

    overall_run_counter = 0
    # SKIP_N_RUNS = 4
    for model_key, config in MODEL_CONFIGS.items():
        model_pt = config["pt_file"]
        model_batch = config["batch_size"]
        model_note = config.get("note", "No specific note.")

        print(f"\n=================================================")
        print(f"Preparing to train model type: {model_key} (using {model_pt})")
        print(f"Configured Batch Size: {model_batch}. Note: {model_note}")
        print(f"=================================================")

        for i, param_set_config in enumerate(PARAMETER_GRID_ADVANCED):
            
            # if overall_run_counter < SKIP_N_RUNS:
            #     param_tag_for_skip_log = param_set_config.get('run_tag', f'param_index_{i}')
            #     print(f"INFO: [{overall_run_counter + 1}/{SKIP_N_RUNS}] Skipping previously completed/flagged run for model '{model_key}', params '{param_tag_for_skip_log}'.")
            #     overall_run_counter += 1
            #     continue
                
            run_training(
                model_name_key=model_key,
                model_pt_file=model_pt,
                model_batch_size=model_batch,
                run_index_overall=overall_run_counter,
                params_from_grid=param_set_config # Pass the whole dict
            )
            overall_run_counter += 1

    print("\nAll configured training runs have been attempted.")