import torch
from efficientvit.cls_model_zoo import get_cls_model

def filter_and_load_weights(model : torch.nn.Module, state_dict):
    new_state_dict = {}
    for key, val in state_dict.items():
        if key in model.state_dict().keys() and val.shape == model.state_dict()[key].shape:
            new_state_dict[key] = val

    return model.load_state_dict(new_state_dict, strict=False)

def get_efficientvit_b1_cls_weights(in_channels, n_classes, weight_path="pretrained_weights/b1-r224.pt", load_weight=True):
    efficientvit = get_cls_model(name="b1", in_channels=in_channels, n_classes=n_classes)
    if load_weight:
        weight = torch.load(weight_path)['state_dict']
        msg = filter_and_load_weights(efficientvit, weight)
        print(msg)
    return efficientvit 