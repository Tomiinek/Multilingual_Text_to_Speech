import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler
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


class SubsetSampler(Sampler):
    """Samples elements sequentially from a given list of indices.

    Arguments:
        indices -- a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in len(self.indices))

    def __len__(self):
        return len(self.indices)


class PerfectBatchSampler(Sampler):
    """Samples a mini-batch of indices for the grouped ConvolutionalEncoder.

    For L samples languages and batch size B produces a mini-batch with
    samples of a particular language L_i (random regardless speaker) 
    on the indices (into the mini-batch) i + k * L for k from 0 to B // L.
    
    Thus can be easily reshaped to a tensor of shape [B // L, L * C, ...]
    with groups consistent with languages.

    Arguments:
        data_source -- dataset to sample from
        languages -- list of languages of data_source to sample from
        batch_size -- total number of samples to be sampled in a mini-batch
        shuffle -- if True, samples randomly, otherwise samples sequentially 

    """

    def __init__(self, data_source, languages, batch_size, shuffle=True):

        assert batch_size % len(languages) == 0, ('Batch size must be divisible by number of languages.')

        lebel_indices = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label not in lebel_indices: lebel_indices[label] = []
            lebel_indices[label].append(idx)

        if shuffle:
            self._samplers = [SubsetRandomSampler(label_indices[i]) for i, _ in enumerate(languages)]
        else:
            self._samplers = [SubsetSampler(label_indices[i]) for i, _ in enumerate(languages)]

        self._batch_size = batch_size

    def __iter__(self):
        batch = []
        for s in self._samplers:
            for idx in s:
                batch.append(idx)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []

    def __len__(self):
        language_batch_size = self._batch_size // len(self._samplers)
        return min(len(s) // language_batch_size) for s in self._samplers)
        