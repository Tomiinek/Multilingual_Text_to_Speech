import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from dataset.dataset import TextToSpeechDataset


class RandomImbalancedSampler(Sampler):
    """Samples randomly imbalanced dataset (with repetition)."""

    def __init__(self, data_source):

        lebel_freq = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label in lebel_freq: lebel_freq[label] += 1
            else: lebel_freq[label] = 1

        total = float(sum(lebel_freq.values()))         
        weights = [total / lebel_freq[data_source.items[idx]['language']] for idx in range(len(data_source))]

        self._sampler = WeightedRandomSampler(weights, len(weights)) 
                
    def __iter__(self):
        return self._sampler.__iter__()

    def __len__(self):
        return len(self._sampler)
