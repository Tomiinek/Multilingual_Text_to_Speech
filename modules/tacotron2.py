import torch
from torch.nn import Sequential, ReLU, Sigmoid, Tanh, Identity, Dropout, Conv1d, BatchNorm1d, LSTM


def get_activation(name):
    """Get activation function by name."""
    return {
        'relu': ReLU()
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'identity': Identity()
    }[name]


class ConvBlock(torch.nn.Module):
    """One dimensional convolution with batchnorm and dropout."""

    def __init__(self, input_channels, output_channels, kernel, dropout, activation='identity'):
        """
        Keyword arguments:
        input_channels -- number if input channels
        output_channels -- number of output channels
        kernel -- convolution kernel size ('same' padding is used)
        dropout -- dropout rate to be aplied after the block
        activation (optional) -- name of the activation function applied after batchnorm (default 'identity')
        """
        super(ConvBlock, self).__init__()
        self._block = Sequential(
            Conv1d(input_channels, output_channels, kernel, padding=(kernel-1)//2, bias=False),
            BatchNorm1d(output_channels),
            get_activation(activation),
            Dropout(dropout)
        )

    def forward(self, x):
        self._block(x)


class Encoder(torch.nn.Module):
    """
    Encoder:
        - stack of 3 conv. layers 5 × 1 with BN and ReLU, dropout
        - output is passed into a Bi-LSTM layer
    """
    
    def __init__(self, input_dim, output_dim, num_blocks, kernel_size, dropout):
        """
        Keyword arguments:
        input_dim -- size of the input (supposed character embedding)
        output_dim -- number of channels of the convolutional blocks and last Bi-LSTM
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
        """
        super(Encoder, self).__init__()
        assert num_blocks > 0, ('There must be at least one convolutional block in the encoder.')
        assert output_dim % 2 == 0, ('Bidirectional LSTM output dimension must be divisible by 2.')
        self._conv1 = ConvBlock(input_dim, output_dim, kernel_size, dropout, 'relu')
        self._convs = [ConvBlock(output_dim, output_dim, kernel_size, dropout, 'relu'))] * (num_blocks - 1)
        self._lstm = LSTM(output_dim, output_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self._conv1(x)
        for conv in self._convs:
            x = conv(x)
        x, _, _ = self.lstm(x)
        return x


class Prenet(torch.nn.Module):
    """
    Prenet:
        - stack of 2 linear layers with dropout which is enabled even during inference (output variation)
        - should act as bottleneck for attention
    """

    def __init__(self, input_dim, output_dim, num_layers, droput):
        """
        Keyword arguments:
        input_dim -- size of the input (supposed the number of frame mels)
        output_dim -- size of the output, should be lower than input_dim to make a bottleneck
        num_layers -- number of the linear layers (at least one)
        dropout -- dropout rate to be aplied after each layer (even during inference)
        """
        super(Prenet, self).__init__()
        assert num_layers > 0, ('There must be at least one layer in the pre-net.')
        self._dropout_rate = dropout
        self._activation = get_activation('relu')
        self._layer1 = Linear(input_dim, output_dim)
        self._layers = [Linear(output_dim, output_dim)] * (num_layers - 1)

    def _layer_pass(self, x, layer):
        x = layer(x)
        x = self._activation(x)
        x = F.dropout(x, p=self._dropout_rate, training=True)
        return x

    def forward(self, x):
        x = self._layer_pass(x, self._layer1)
        for layer in self._layers:
            x = self._layer_pass(x, layer)
        return x


class Postnet(torch.nn.Module):
    """
    Postnet:
        - stack of 5 conv. layers 5 × 1 with BN and tanh (except last), dropout
    """

    def __init__(self, dimension, num_blocks, kernel_size, dropout):
        """
        Keyword arguments:
        dimension -- size of the input and output (supposed the number of frame mels)
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
        """
        super(Postnet, self).__init__()
        assert num_blocks > 0, ('There must be at least one convolutional block in the post-net.')
        self._conv_last = ConvBlock(dimension, dimension, kernel_size, dropout, 'identity')
        self._convs = [ConvBlock(dimension, dimension, kernel_size, dropout, 'tanh'))] * (num_blocks - 1)

    def forward(self, x):
        for conv in self._convs:
            x = conv(x)
        x = self._conv_last(x)
        return x


class Attention(torch.nn.Module):
    """
    Attention:
        - location-sensitive attention: https://arxiv.org/abs/1506.07503
        - which extends additive attention: https://arxiv.org/abs/1409.0473
          to use !cumulative attention weights! from previous decoder time steps
        - attention probabilities are computed after projecting inputs and 
          location features to 128-dimensional hidden representations
    Notes on weights: 
        - weights_{i} = Attend(state_{i−1} , cum_weights_{i−1} , encoder_outputs)
        - Attend:
          conv_features_{i} = F ∗ cum_weights_{i−1}
          energy_{i,j} = w^T tanh(W state_{i−1} + V encoder_outputs_{j} + U conv_features_{i,j} + b)
          the following normalization might use sigmoid instead of exp and some kind of windowing
          FIXME: however I do not undersand the paper and do not know how should the window look like 
          FIXME: I skip any windowing or sharpening as described in the paper, but should be tried!!!
          weights_{i} = softmax(energy_{i})
    """

    def __init__(self, representation_dim, query_dim, memory_dim, kernel_size, channels, smoothing=False):
        """
        Keyword arguments:
        representation_dim -- size of the hidden representation
        query_dim -- size of the attention query input (probably decoder hidden state)
        memory_dim -- size of the attention memory input (probably encoder outputs) 
        kernel_size -- kernel size of the convolution calculating location features 
        channels -- number of channels of the convolution calculating location features 
        smoothing -- to normalize weights using softmax, use False (default) and True to use sigmoids 
        """
        super(Attention, self).__init__()
        self._bias = Parameter(torch.zeros(1, representation_dim))
        self._energy = Linear(representation_dim, 1, bias=False)     
        self._query = Linear(query_dim, representation_dim, bias=False)          
        self._memory = Linear(memory_dim, representation_dim, bias=False)
        self._location = Linear(channels, representation_dim, bias=False)
        self._loc_features = Conv1d(1, channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.smoothing = smoothing

    def get_memory_layer(self):
        """Used to pre-compute memory, as it is independent from decoder steps."""
        return self._memory

    def _attend(self, query, memory_transform, cum_weights):
        query = query.unsqueeze(1)
        query = self._query(query)
        cum_weights = cum_weights.unsqueeze(-1)
        loc_features = self._loc_features(cum_weights)
        loc_features = self._location(loc_features)
        energy = query + memory_transform + loc_features
        energy = self._energy(F.tanh(energy + self._bias))
        return energy.squeeze(-1)

    def _normalize(self, energies, mask):
        energies[~mask] = float('-inf')
        if self.smoothing:
            sigmoid = F.sigmoid(energies)
            total = torch.sum(sigmoid, dim=-1)
            return sigmoid / total
        else:
            return F.softmax(energies, dim=1)

    def forward(self, query, memory, memory_transform, cum_weights, mask):
        energies = self._attend(query, memory_transform, cum_weights)
        attention_weights = self._normalize(energies, mask)
        attention_weights = attention_weights.unsqueeze(1)
        context = torch.bmm(attention_weights, memory)
        return context.squeeze(1), attention_weights


class Decoder(torch.nn.Module):
    """
    Decoder:
        - stack of 2 uni-directional LSTM layers with 1024 units
        - first LSTM is used to query attention mechanism
        - input of the first LSTM is previous prediction (pre-net output) and previous context vector
        - second LSTM acts as a generator
        - input of the second LSTM is current context vector and output of the first LSTM
        - output is passed through stop token layer, frame prediction layer and pre-net
    """

    def __init__(self, output_dim, decoder_dim, attention, context_dim, prenet, prenet_dim):
        """
        Keyword arguments:
        output_dim -- size of the predicted frame, i.e. number of mels 
        decoder_dim -- size of the decoder output (and also of all the LSTMs used in the decoder)
        attention -- instance of the location-sensitive attention module 
        context_dim -- size of the context vector produced by the given attention
        prenet -- instance of the pre-net module 
        prenet_dim -- output dimension of the pre-net
        """
        super(Decoder, self).__init__()
        self.prenet = prenet
        self.attention = attention
        self.attention_lstm = LSTMCell(context_dim + prenet_dim, decoder_dim, batch_first=True) 
        self.generator_lstm = LSTMCell(context_dim + decoder_dim, decoder_dim, batch_first=True)
        self.frame_prediction = Linear(context_dim + decoder_dim, output_dim)
        self.stop_prediction = Linear(context_dim + decoder_dim, 1)

    def prepare_target(self, target):

        # TODO: prepend zeros to target
        target = self.prenet(target)
        return target

    def forward(self, target, encoded_input, mask):       
        
         target = self.prepare_target(target)
        
        # TODO: prepare initial states, perform decoding
        for _ in range(target.size(0) - 1):
            self.decoder_step()

        return spectrogram, stop_tokens, alignments

    def inference(self, encoded_input, mask):
        # perform decoding
        return spectrogram, stop_tokens, alignments
    
    







class Tacotron(torch.nn.Module):
    """
    Tacotron 2:
    - previous time step passed through pre-net (2 FC of 256 ReLU)
    - pre-net output and attention context are concatenated into input
    - LSTM output and attention context are concatenated into output
        - characters as learned 512-dimensional embedding
        - the predicted mel spectrogram is passed through post-net which
          predicts a residual to add to the prediction
        - minimize MSE from before and after the post-net to aid convergence
        - parallel to spectrogram frame prediction, the concatenation of
          decoder LSTM output and attention context is projected down to a scalar 
          and passed through sigmoid to predict “stop token”
        - convolutional layers in the network are regularized using dropout
          with probability 0.5, and LSTM layers are regularized using zoneout 
          with probability 0.1
        - we do not use a “reduction factor” <- (definitely not together with wavenet!)
        - TODO: mask -- maxlen = X.size(1)
                        mask = torch.arange(maxlen)[None, :] < X_len[:, None]
        - TODO: make it possible to use cumulative attention weights
        - TODO: zoneout https://github.com/WelkinYang/Zoneout-Pytorch/blob/master/ZoneoutRNN.py
        - TODO: gradient clipping?
        - TODO: guided attention loss
        - TODO: Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
        - TODO: Multi-headed attention? Neural Speech Synthesis with Transformer Network, already implemented somewhere :(
    """

    def __init__(self):
        # TODO: create attention
        # TOOD: create encoder
        # TODO: create prenet
        # TODO: create decoder
        # TODO: create postnet
        pass

    def forward(self, x):
        pass