import torch


def discover_devices():
    """Check for available devices."""
    return "cuda" if torch.cuda.is_available() else "cpu"
