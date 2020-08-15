import numpy as np

# Helper function to convert a torch tensor to NumPy for plotting
def tensor_to_numpy(t):
    return np.clip(t.numpy().transpose((1, 2, 0)), 0, 1)
