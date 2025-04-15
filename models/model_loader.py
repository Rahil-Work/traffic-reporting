# project/models/model_loader.py
import torch
from ultralytics import YOLO
from core.utils import debug_print

def load_model(model_path, device, conf_threshold, iou_threshold):
    """Loads and configures the YOLO model."""
    debug_print(f"Loading model from: {model_path} onto device: {device}")
    try:
        model = YOLO(model_path)
        model.to(device)
        model.conf = conf_threshold
        model.iou = iou_threshold
        model.verbose = False
        debug_print(f"Model loaded. conf={model.conf}, iou={model.iou}")

        # Optional warm-up
        if device == 'cuda':
            try:
                debug_print("Warming up model...")
                inp_size = getattr(model, 'imgsz', 640)
                if isinstance(inp_size, list): inp_size = inp_size[0]
                dummy_input = torch.zeros(1, 3, inp_size, inp_size).to(device)
                model(dummy_input, verbose=False)
                torch.cuda.synchronize()
                debug_print("Model warm-up complete.")
            except Exception as e: debug_print(f"Warn: Warm-up failed: {e}")
        return model
    except Exception as e: print(f"FATAL ERROR loading model '{model_path}': {e}"); raise

def get_device():
    """Determines the best available device."""
    if torch.cuda.is_available(): device = 'cuda'; print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'; print("Using MPS") # Basic MPS check
    else: device = 'cpu'; print("CUDA not available: Using CPU")
    return device