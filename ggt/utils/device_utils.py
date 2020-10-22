import torch
import logging

def discover_devices():
    """Check for available devices."""
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = (torch.cuda.get_device_name(i) for i in range(n_devices))
        logging.info(f"Using {n_devices} GPUs {devices}")
        return "cuda"
    else:
        logging.info("No GPU found; falling back to CPU")
        return "cpu"
