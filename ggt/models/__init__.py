from .ggt_model import GGT


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable_params=n_params)


def model_factory(modeltype):
    if modeltype.lower() == 'ggt':
        return GGT
    else:
        raise ValueError("Model type {} does not exist.".format(modeltype))
