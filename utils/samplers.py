import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler
from dataset.dataset import TextToSpeechDataset


class RandomImbalancedSampler(Sampler):
    """Samples randomly imbalanced dataset (with repetition).
    
    Argument:
        data_source -- instance of TextToSpeechDataset
    """

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
        return (self.indices[i] for i in range(len(self.indices)))

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
        data_parallel_devices -- number of parallel devices used in the data parallel mode which splits batch as we need
                                 to ensure that B (or smaller batch if drop_last is False) is divisible by (L * this_argument) 
        shuffle -- if True, samples randomly, otherwise samples sequentially 
        drop_last -- if True, drops last imcomplete batch
    """

    def __init__(self, data_source, languages, batch_size, data_parallel_devices=1, shuffle=True, drop_last=False):

        assert batch_size % (len(languages) * data_parallel_devices) == 0, (
            'Batch size must be divisible by number of languages times the number of data parallel devices (if enabled).')

        label_indices = {}
        for idx in range(len(data_source)):
            label = data_source.items[idx]['language']
            if label not in label_indices: label_indices[label] = []
            label_indices[label].append(idx)

        if shuffle:
            self._samplers = [SubsetRandomSampler(label_indices[i]) for i, _ in enumerate(languages)]
        else:
            self._samplers = [SubsetSampler(label_indices[i]) for i, _ in enumerate(languages)]

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dp_devices = data_parallel_devices

    def __iter__(self):
        
        batch = []
        iters = [iter(s) for s in self._samplers]
        done = False
        
        while True:
            b = []
            for it in iters:
                idx = next(it, None)
                if idx is None:
                    done = True
                    break
                b.append(idx)
            if done: break
            batch += b
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        
        if not self._drop_last:
            if len(batch) > 0:
                groups = len(batch) // len(self._samplers)
                if groups % self._dp_devices == 0:
                    yield batch
                else:
                    batch = batch[:(groups // self._dp_devices) * self._dp_devices * len(self._samplers)]
                    if len(batch) > 0:
                        yield batch 
        
    def __len__(self):
        language_batch_size = self._batch_size // len(self._samplers)
        return min(((len(s) + language_batch_size - 1) // language_batch_size) for s in self._samplers)