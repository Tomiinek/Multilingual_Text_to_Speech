import torch


def lengths_to_mask(lengths, max_length=None):
    ml = torch.max(lengths) if max_length is None else max_length
    return torch.arange(ml, device=lengths.device)[None, :] < lengths[:, None]