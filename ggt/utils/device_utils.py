import torch


def discover_devices():
    return "cuda" if torch.cuda.is_available() else "cpu"
