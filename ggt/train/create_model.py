import torch
import torch.nn as nn

from ggt.models import model_factory

def create_model(model_name, mode, model_type, device, saved_state=None):
    logging.info("Building model {} ({})...".format(model_name, model_type))
    cls = model_factory(model_type)
    model = cls(mode=mode)
    model = nn.DataParallel(model)
    model = model.to(device)

    if saved_state:
        logging.info(" -- loading state from {}...".format(saved_state))
        model.load_state_dict(torch.load(saved_state))

    utils.log_dictionary(logging.info, model_stats(model))
    print(model)
    return model
