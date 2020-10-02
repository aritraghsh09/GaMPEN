import torch
import logging

def discover_devices():
    """Check for available devices."""
    if torch.cuda.is_available():

        logging.info("Using the following GPUs")

        for i in range(0,torch.cuda.device_count()):
            logging.info(torch.cuda.get_device_name(i))

        return "cuda"

    else:
        logging.info("GPU not found! Using a CPU")
        return "cpu"
