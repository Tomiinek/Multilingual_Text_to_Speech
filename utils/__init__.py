import torch
from collections import OrderedDict
from params.params import Params as hp
from modules.tacotron2 import Tacotron


def lengths_to_mask(lengths, max_length=None):
    """Convert tensor of lengths into a boolean mask."""
    ml = torch.max(lengths) if max_length is None else max_length
    return torch.arange(ml, device=lengths.device)[None, :] < lengths[:, None]


def to_gpu(x):
    """Compact and move CPU tensor to GPU."""
    if x is None: return x
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def remove_dataparallel_prefix(state_dict): 
    """Removes dataparallel prefix of layer names in a checkpoint state dictionary."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:7] == "module." else k
        new_state_dict[name] = v
    return new_state_dict


def build_model(checkpoint, force_cpu=False):   
    """Load and build model a from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    state = torch.load(checkpoint, map_location=device)
    hp.load_state_dict(state['parameters'])
    model = Tacotron()
    model.load_state_dict(remove_dataparallel_prefix(state['model']))   
    model.to(device)
    return model
