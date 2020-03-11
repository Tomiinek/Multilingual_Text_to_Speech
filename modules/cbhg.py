import torch
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, Linear, ReLU, Sigmoid, MaxPool1d, ConstantPad1d, MaxPool1d, GRU

from modules.layers import ConvBlock


class PostnetCBHG(torch.nn.Module):
    """
    CBHG block with a linear output layer.

    Arguments:
        input_dim -- number of channels of the input (probably should match the number of mels)
        output_dim -- number of channels of the output
        bank_size -- number of convolutions with kernel 1..bank_size in the convolutional bank
        bank_channels -- output channels of convolutions in the bank
        projection_channels -- channels of the convolutional projection layers (these layers are two)
        projection_kernel_size -- kernel size of the convolutional projection layers      
        highway_dim -- output dimension of the highway layers
        gru_dim -- dim of the GRU layer
        dropout -- dropout of the convolution layers
    """

    def __init__(self, input_dim, output_dim, bank_size, bank_channels, projection_channels, projection_kernel_size, highway_dim, gru_dim, dropout):
        super(PostnetCBHG, self).__init__()
        assert gru_dim % 2 == 0, ('Bidirectional GRU dimension must be divisible by 2.')
        self._bank = ModuleList([
            ConvBlock(input_dim, bank_channels, k, dropout, 'relu') for k in range(1, bank_size + 1)
        ])
        self._pool_and_project = Sequential(
            ConstantPad1d((0, 1), 0.0),
            MaxPool1d(2, stride=1),
            ConvBlock(bank_channels * bank_size, projection_channels, projection_kernel_size, dropout, 'relu'),
            ConvBlock(projection_channels, input_dim, projection_kernel_size, dropout, 'identity')
        )
        highways = [HighwayLayer(highway_dim) for _ in range(4)]
        self._highway_layers = Sequential(
            Linear(input_dim, highway_dim),
            ReLU(),
            *highways
        )
        self._gru = GRU(highway_dim, gru_dim // 2, batch_first=True, bidirectional=True)
        self._output_layer = Linear(gru_dim, output_dim)
    
    def forward(self, x, x_lengths):
        
        residual = x
        bx = [layer(x) for layer in self._bank]
        x = torch.cat(bx, dim=1)
        x = self._pool_and_project(x)
        
        x = x + residual
        x = x.transpose(1, 2)
        x = self._highway_layers(x)
        
        ml = x.size(1)
        sorted_lengths, sorted_idxs = torch.sort(x_lengths, descending=True)
        x = x[sorted_idxs]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
        self._gru.flatten_parameters()
        x, _ = self._gru(x)
        bx, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=ml) 
        x = torch.zeros_like(bx)
        x[sorted_idxs] = bx

        x = self._output_layer(x)
        x = x.transpose(1, 2)
        
        return x


class HighwayLayer(torch.nn.Module):
    """Gated layer."""

    def __init__(self, dimension):
        super(HighwayLayer, self).__init__()
        self._linear = Sequential(
            Linear(dimension, dimension),
            ReLU()
        )
        self._gate = Sequential(
            Linear(dimension, dimension),
            Sigmoid()
        )

    def forward(self, x):
        p = self._gate(x)
        return self._linear(x) * p + x * (1.0 - p)