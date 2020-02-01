import torch
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, ReLU, LSTM, Embedding

from modules.layers import ConvBlock, HighwayConvBlock, ConvBlockGenerated, HighwayConvBlockGenerated
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
        convs = [ConvBlock(input_dim, output_dim, kernel_size, dropout, 'relu')] + \
                [ConvBlock(output_dim, output_dim, kernel_size, dropout, 'relu') for _ in range(num_blocks - 1)]
        self._convs = Sequential(*convs)
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
    """
    Convolutional encoder (possibly multi-lingual).

    Expects input of shape [B * N, L, F], where B is divisible by N (number of languages) and
    samples of each language with the first sample at the i-th position occupy every i+L-th 
    position in the batch (so that it can be reshaped to [B, N * F, L] easily).

    Arguments:
        input_dim -- size of the input (supposed character embedding)
        output_dim -- number of channels of the convolutional blocks and output
        dropout -- dropout rate to be aplied after each convolutional block
    """

    def __init__(self, input_dim, output_dim, dropout, groups=1):
        super(ConvolutionalEncoder, self).__init__()

        self._groups = groups
        self._input_dim = input_dim
        self._output_dim = output_dim

        input_dim *= groups
        output_dim *= groups 

        layers = [ConvBlock(input_dim, output_dim, 1, dropout, activation='relu', groups=groups),
                  ConvBlock(output_dim, output_dim, 1, dropout, groups=groups)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlock(output_dim, output_dim, 3, dropout, dilation=1, groups=groups) for _ in range(2)] + \
                 [HighwayConvBlock(output_dim, output_dim, 1, dropout, dilation=1, groups=groups) for _ in range(2)]
        
        self._layers = Sequential(*layers)

    def forward(self, x, x_lenghts=None, x_langs=None):

        # x_langs is specified during inference with batch size 1, so we need to 
        # pad the single language with zeros to create complete groups
        if x_langs is not None and x_langs.shape[0] == 1:
            x_e = torch.zeros([self._groups, x.shape[1], x.shape[2]], device=x.device)
            x_e[x_langs] = x
            x = x_e

        bs = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(bs // self._groups, self._groups * self._input_dim, -1)
        x = self._layers(x)
        x = x.reshape(bs, self._output_dim, -1)
        x = x.transpose(1, 2)

        if x_langs is not None and x_langs.shape[0] == 1:
            x = x[x_langs]

        return x


class GeneratedConvolutionalEncoder(torch.nn.Module):
    """Convolutional encoder (possibly multi-lingual) with weights generated by another network.
    
    Arguments:
        see ConvolutionalEncoder
        embedding_dim -- size of the generator embedding (should be language embedding)
        bottleneck_dim -- size of the layer between generated weights and generator embedding
    """

    def __init__(self, input_dim, output_dim, dropout, embedding_dim, bottleneck_dim, groups=1):
        super(GeneratedConvolutionalEncoder, self).__init__()
        
        self._groups = groups
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        input_dim *= groups
        output_dim *= groups
        
        layers = [ConvBlockGenerated(embedding_dim, bottleneck_dim, input_dim, output_dim, 1,
                                     dropout=dropout, activation='relu', groups=groups),
                  ConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 1,
                                     dropout=dropout, groups=groups)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3, 
                                            dropout=dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3,
                                            dropout=dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3,
                                            dropout=dropout, dilation=1, groups=groups) for _ in range(2)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 1,
                                            dropout=dropout, dilation=1, groups=groups) for _ in range(2)]
        
        self._layers = Sequential(*layers)
        self._embedding = Embedding(groups, embedding_dim)

    def forward(self, x, x_lenghts=None, x_langs=None):

        # x_langs is specified during inference with batch size 1, so we need to 
        # pad the single language with zeros to create complete groups
        if x_langs is not None and x_langs.shape[0] == 1:
            x_e = torch.zeros([self._groups, x.shape[1], x.shape[2]], device=x.device)
            x_e[x_langs] = x
            x = x_e

        # create generator embeddings for all groups
        e = self._embedding(torch.arange(self._groups, device=x.device))

        bs = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(bs // self._groups, self._groups * self._input_dim, -1)   
        _, x = self._layers((e, x))
        x = x.reshape(bs, self._output_dim, -1)
        x = x.transpose(1, 2)

        if x_langs is not None and x_langs.shape[0] == 1:
            x = x[x_langs]

        return x
