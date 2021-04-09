import torch


def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


def enable_dropout(model):
    """Enable random dropout during inference. From StackOverflow #63397197"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
