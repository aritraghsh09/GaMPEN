import torch
from pathlib import Path

from .ggt import GGT
from .ggt_no_gcov import GGT_no_gconv
from .vgg import vgg16
from .vgg16_w_stn_drp import vgg16_w_stn_drp
from .vgg16_w_stn_drp_2 import vgg16_w_stn_drp_2
from .vgg16_w_stn_at_drp import vgg16_w_stn_at_drp
from .vgg16_w_stn_oc_drp import vgg16_w_stn_oc_drp


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable_params=n_params)


def model_factory(modeltype):
    if modeltype.lower() == "ggt":
        return GGT
    elif modeltype.lower() == "ggt_no_gconv":
        return GGTNoGConv
    elif modeltype.lower() == "vgg16":
        return vgg16
    elif modeltype.lower() == "ggt_no_gconv":
        return GGT_no_gconv
    elif (
        modeltype.lower() == "vgg16_w_stn_drp"
        or modeltype.lower() == "vgg16_w_stn"
    ):
        return vgg16_w_stn_drp
    elif modeltype.lower() == "vgg16_w_stn_drp_2":
        return vgg16_w_stn_drp_2
    elif modeltype.lower() == "vgg16_w_stn_at_drp":
        return vgg16_w_stn_at_drp
    elif modeltype.lower() == "vgg16_w_stn_oc_drp":
        return vgg16_w_stn_oc_drp
    else:
        raise ValueError("Model type {} does not exist.".format(modeltype))


def save_trained_model(model, slug):
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{slug}.pt"
    torch.save(model.state_dict(), dest)
    return dest
