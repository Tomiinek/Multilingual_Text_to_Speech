import torch
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, ReLU, LSTM, Embedding

from modules.layers import ConvBlock, HighwayConvBlock
from modules.generated import LSTMGenerated
from params.params import Params as hp


class Encoder(torch.nn.Module):
    """
    Encoder:
        stack of 3 conv. layers 5 × 1 with BN and ReLU, dropout
        output is passed into a Bi-LSTM layer

    Arguments:
        input_dim -- size of the input (supposed character embedding)
        output_dim -- number of channels of the convolutional blocks and last Bi-LSTM
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
        generated -- enables meta-learning approach which generates parameters of the internal layers
    """
    
    def __init__(self, input_dim, output_dim, num_blocks, kernel_size, dropout, generated=False):
        super(Encoder, self).__init__()
        assert num_blocks > 0, ('There must be at least one convolutional block in the encoder.')
        assert output_dim % 2 == 0, ('Bidirectional LSTM output dimension must be divisible by 2.')
        convs = [ConvBlock(input_dim, output_dim, kernel_size, dropout, 'relu', generated)] + \
                [ConvBlock(output_dim, output_dim, kernel_size, dropout, 'relu', generated) for _ in range(num_blocks - 1)]
        self._convs = Sequential(*convs)
        if generated:
            self._lstm = LSTMGenerated(output_dim, output_dim // 2, batch_first=True, bidirectional=True)
        else:
            self._lstm = LSTM(output_dim, output_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x, x_lenghts, x_langs=None):  
        # x_langs argument is there just for convenience
        x = x.transpose(1, 2)
        x = self._convs(x)
        x = x.transpose(1, 2)
        ml = x.size(1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts, batch_first=True)
        self._lstm.flatten_parameters()
        x, _ = self._lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=ml) 
        return x


class ConditionalEncoder(torch.nn.Module):
    """
    Encoder which has a language embeddings concatenated to each input character embedding.

    Arguments:
        input_dim -- total number of languages in the dataset
        langs_embedding_dim -- output size of the language embedding
        encoder_args -- tuple or list of arguments for encoder (input dimension without the embedding dimension)

    TODO: conditioning after convolutional layers?
    TODO: larger dimension of the other conv. layers or of the LSTM? 
    """
    
    def __init__(self, num_langs, langs_embedding_dim, encoder_args):
        super(ConditionalEncoder, self).__init__()
        self._language_embedding = Embedding(num_langs, langs_embedding_dim)
        # modify input_dim of the underlying Encoder
        encoder_args = list(encoder_args)
        encoder_args[0] += langs_embedding_dim
        encoder_args = tuple(encoder_args)
        self._encoder = Encoder(*encoder_args)

    def forward(self, x, x_lenghts, x_langs):  
        l = self._language_embedding(x_langs)
        expanded_l = l.unsqueeze(1).expand((-1, x.shape[1], -1))
        x = torch.cat((x, expanded_l), dim=-1) 
        x = self._encoder(x, x_lenghts)
        return x


class MultiEncoder(torch.nn.Module):
    """
    Bunch of language-dependent encoders with output masking.

    Arguments:
        num_langs -- number of languages (and encoders to be instiantiated)
        encoder_args -- tuple or list of arguments for encoder
    """

    def __init__(self, num_langs, encoder_args):
        super(MultiEncoder, self).__init__()
        self._num_langs = num_langs
        self._encoders = ModuleList([Encoder(*encoder_args) for _ in range(num_langs)])

    def forward(self, x, x_lenghts, x_langs):
        xs = None
        for l in range(self._num_langs):
            mask = (x_langs == l)
            if not mask.any(): continue
            ex = self._encoders[l](x, x_lenghts)
            if xs is None:
                xs = torch.zeros_like(ex)
            xs[mask] = ex[mask]
        return xs


class ConvolutionalEncoder(torch.nn.Module):

    def __init__(self, input_dim, output_dim, dropout, generated=False):
        super(ConvolutionalEncoder, self).__init__()
        layers = [ConvBlock(input_dim, output_dim, 1, dropout, activation='relu'),
                  ConvBlock(output_dim, output_dim, 1, dropout)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=3**i) for i in range(4)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=3**i) for i in range(4)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=1) for _ in range(2)] + \
                 [HighwayConvBlock(output_dim, output_dim, 1, dropout, dilation=1) for _ in range(2)]
        self._layers = Sequential(*layers)

    def forward(self, x, x_lenghts=None, x_langs=None):
        x = x.transpose(1, 2)
        x = self._layers(x)
        x = x.transpose(1, 2)
        return x
