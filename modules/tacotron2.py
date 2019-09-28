import torch
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, Linear, ReLU, Sigmoid, Tanh, Identity, Dropout, Conv1d, BatchNorm1d, Parameter, LSTM, LSTMCell, Embedding

from params.params import Params as hp


def get_activation(name):
    """Get activation function by name."""
    return {
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'identity': Identity()
    }[name]


def lengths_to_mask(lengths):
    return torch.arange(torch.max(lengths))[None, :] < lengths[:, None]


class ZoneoutLSTMCell(torch.nn.LSTMCell):
    """Wrapper around LSTM cell providing zoneout regularization."""

    def __init__(self, input_size, hidden_size, zoneout_rate_hidden, zoneout_rate_cell, bias=True):
        super(ZoneoutLSTMCell, self).__init__(input_size, hidden_size, bias)
        self.zoneout_c = zoneout_rate_cell
        self.zoneout_h = zoneout_rate_hidden

    def forward(self, cell_input, h, c):
        new_h, new_c = super(ZoneoutLSTMCell, self).forward(cell_input, (h, c))
        if self.training:
            new_h = (1-self.zoneout_h) * F.dropout(new_h - h, self.zoneout_h) + h
            new_c = (1-self.zoneout_c) * F.dropout(new_c - c, self.zoneout_c) + c
        else:
            new_h = self.zoneout_h * h + (1-self.zoneout_h) * new_h
            new_c = self.zoneout_c * c + (1-self.zoneout_c) * new_c
        return new_h, new_c


class ConvBlock(torch.nn.Module):
    """One dimensional convolution with batchnorm and dropout, expected channel-first input.
    
    Keyword arguments:
    input_channels -- number if input channels
    output_channels -- number of output channels
    kernel -- convolution kernel size ('same' padding is used)
    dropout -- dropout rate to be aplied after the block
    activation (optional) -- name of the activation function applied after batchnorm (default 'identity')
    """

    def __init__(self, input_channels, output_channels, kernel, dropout, activation='identity'):
        super(ConvBlock, self).__init__()
        self._block = Sequential(
            Conv1d(input_channels, output_channels, kernel, padding=(kernel-1)//2, bias=False),
            BatchNorm1d(output_channels),
            get_activation(activation),
            Dropout(dropout)
        )

    def forward(self, x):
        return self._block(x)


class Encoder(torch.nn.Module):
    """
    Encoder:
        stack of 3 conv. layers 5 × 1 with BN and ReLU, dropout
        output is passed into a Bi-LSTM layer

    Keyword arguments:
        input_dim -- size of the input (supposed character embedding)
        output_dim -- number of channels of the convolutional blocks and last Bi-LSTM
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
    """
    
    def __init__(self, input_dim, output_dim, num_blocks, kernel_size, dropout):
        super(Encoder, self).__init__()
        assert num_blocks > 0, ('There must be at least one convolutional block in the encoder.')
        assert output_dim % 2 == 0, ('Bidirectional LSTM output dimension must be divisible by 2.')
        self._convs = Sequential(
            ConvBlock(input_dim, output_dim, kernel_size, dropout, 'relu'),
            *[ConvBlock(output_dim, output_dim, kernel_size, dropout, 'relu')] * (num_blocks - 1)
        )
        self._lstm = LSTM(output_dim, output_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x, x_lenghts):  
        x = x.transpose(1, 2)
        x = self._convs(x)
        x = x.transpose(1, 2)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts, batch_first=True)
        x, _ = self._lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True) 
        return x


class Prenet(torch.nn.Module):
    """
    Prenet:
        stack of 2 linear layers with dropout which is enabled even during inference (output variation)
        should act as bottleneck for attention

    Keyword arguments:
        input_dim -- size of the input (supposed the number of frame mels)
        output_dim -- size of the output, should be lower than input_dim to make a bottleneck
        num_layers -- number of the linear layers (at least one)
        dropout -- dropout rate to be aplied after each layer (even during inference)
    """

    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(Prenet, self).__init__()
        assert num_layers > 0, ('There must be at least one layer in the pre-net.')
        self._dropout_rate = dropout
        self._activation = get_activation('relu')
        self._layers = ModuleList([Linear(input_dim, output_dim)] + [Linear(output_dim, output_dim)] * (num_layers - 1))

    def _layer_pass(self, x, layer):
        x = layer(x)
        x = self._activation(x)
        x = F.dropout(x, p=self._dropout_rate, training=True)
        return x

    def forward(self, x):
        for layer in self._layers:
            x = self._layer_pass(x, layer)
        return x


class Postnet(torch.nn.Module):
    """
    Postnet:
        stack of 5 conv. layers 5 × 1 with BN and tanh (except last), dropout

    Keyword arguments:
        input_dimension -- size of the input and output (supposed the number of frame mels)
        postnet_dimension -- size of the internal convolutional blocks
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
    """

    def __init__(self, input_dimension, postnet_dimension, num_blocks, kernel_size, dropout):
        super(Postnet, self).__init__()
        assert num_blocks > 1, ('There must be at least one convolutional block in the post-net.')
        self._convs = Sequential(
            ConvBlock(input_dimension, postnet_dimension, kernel_size, dropout, 'tanh'),
            *[ConvBlock(postnet_dimension, postnet_dimension, kernel_size, dropout, 'tanh')] * (num_blocks - 2),
            ConvBlock(postnet_dimension, dimension, kernel_size, dropout, 'identity')
        )

    def forward(self, x):
        x = self._convs(x)
        return x


class Attention(torch.nn.Module):
    """
    Attention:
        location-sensitive attention: https://arxiv.org/abs/1506.07503
        which extends additive attention: https://arxiv.org/abs/1409.0473
          to use !cumulative attention weights! from previous decoder time steps
        attention probabilities are computed after projecting inputs and 
          location features to 128-dimensional hidden representations
        
    Keyword arguments:
        representation_dim -- size of the hidden representation
        query_dim -- size of the attention query input (probably decoder hidden state)
        memory_dim -- size of the attention memory input (probably encoder outputs) 
        kernel_size -- kernel size of the convolution calculating location features 
        channels -- number of channels of the convolution calculating location features 
        smoothing -- to normalize weights using softmax, use False (default) and True to use sigmoids 

    Notes on weights: 
        - weights_{i} = Attend(state_{i−1} , cum_weights_{i−1} , encoder_outputs)
        - Attend:
          conv_features_{i} = F ∗ cum_weights_{i−1}
          energy_{i,j} = w^T tanh(W state_{i−1} + V encoder_outputs_{j} + U conv_features_{i,j} + b)
          the following normalization might use sigmoid instead of exp and some kind of windowing
          FIXME: however I do not undersand the paper and do not know how should the window look like 
          FIXME: I skip any windowing or sharpening as described in the paper, but should be tried!!!
          weights_{i} = softmax(energy_{i}
    """

    def __init__(self, representation_dim, query_dim, memory_dim, kernel_size, channels, smoothing=False):
        super(Attention, self).__init__()
        self._bias = Parameter(torch.zeros(1, representation_dim))
        self._energy = Linear(representation_dim, 1, bias=False)     
        self._query = Linear(query_dim, representation_dim, bias=False)          
        self._memory = Linear(memory_dim, representation_dim, bias=False)
        self._location = Linear(channels, representation_dim, bias=False)
        self._loc_features = Conv1d(1, channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self._smoothing = smoothing

    def get_memory_layer(self):
        """Used to pre-compute memory, as it is independent from decoder steps."""
        return self._memory

    def _attend(self, query, memory_transform, cum_weights):
        query = self._query(query.unsqueeze(1))
        cum_weights = cum_weights.unsqueeze(-1)
        loc_features = self._loc_features(cum_weights.transpose(1,2))
        loc_features = self._location(loc_features.transpose(1,2))
        energy = query + memory_transform + loc_features
        energy = self._energy(torch.tanh(energy + self._bias))
        return energy.squeeze(-1)

    def _normalize(self, energies, mask):
        energies[~mask] = float('-inf')
        if self._smoothing:
            sigmoid = torch.sigmoid(energies)
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
        stack of 2 uni-directional LSTM layers with 1024 units
        first LSTM is used to query attention mechanism
        input of the first LSTM is previous prediction (pre-net output) and previous context vector
        second LSTM acts as a generator
        input of the second LSTM is current context vector and output of the first LSTM
        output is passed through stop token layer, frame prediction layer and pre-ne

    Keyword arguments:
        output_dim -- size of the predicted frame, i.e. number of mels 
        decoder_dim -- size of the generator output (and also of all the LSTMs used in the decoder)
        attention -- instance of the location-sensitive attention module 
        context_dim -- size of the context vector produced by the given attention
        prenet -- instance of the pre-net module 
        prenet_dim -- output dimension of the pre-net
        max_length -- maximal length of the input sequence
        max_frames -- maximal number of the predicted frames
    """

    # TODO: support batch size >1 during inference

    def __init__(self, output_dim, decoder_dim, attention, context_dim, prenet, prenet_dim, max_frames):
        super(Decoder, self).__init__()
        self._prenet = prenet
        self._attention = attention
        self._output_dim = output_dim
        self._decoder_dim = decoder_dim
        self._max_frames = max_frames
        self._context_dim = context_dim
        self._attention_lstm = ZoneoutLSTMCell(context_dim + prenet_dim, decoder_dim, hp.zoneout_hidden, hp.zoneout_cell) 
        self._generator_lstm = ZoneoutLSTMCell(context_dim + decoder_dim, decoder_dim, hp.zoneout_hidden, hp.zoneout_cell)
        self._frame_prediction = Linear(context_dim + decoder_dim, output_dim)
        self._stop_prediction = Linear(context_dim + decoder_dim, 1)

    def _attention_init(self, encoded_input, batch_size, max_len):
        """Initialize context and attention weights & prepare attention memory."""
        memory_transform = self._attention.get_memory_layer()(encoded_input)
        context = torch.zeros(batch_size, self._context_dim)
        weights = torch.zeros(batch_size, max_len)
        return memory_transform, context, weights

    def _target_init(self, target, batch_size):
        """Prepend target spectrogram with a zero frame and pass it through pre-net."""
        # the F.pad function has some issues: https://github.com/pytorch/pytorch/issues/13058
        first_frame = torch.zeros(batch_size, self._output_dim).unsqueeze(1)
        target = target.transpose(1, 2) # [B, F, N_MEL]
        target = torch.cat((first_frame, target), dim=1)
        target = self._prenet(target)
        return target

    def _decoder_init(self, batch_size):
        """Initialize hidden and cell state of the deocder's RNNs."""
        h_att = torch.zeros(batch_size, self._decoder_dim)
        c_att = torch.zeros(batch_size, self._decoder_dim)
        h_gen = torch.zeros(batch_size, self._decoder_dim)
        c_gen = torch.zeros(batch_size, self._decoder_dim)
        return h_att, c_att, h_gen, c_gen

    def _decode(self, encoded_input, mask, target=None):
        """Perform decoding of the encoded input sequence."""
        
        # attention and decoder states initialization
        batch_size = encoded_input.size(0)
        max_length = encoded_input.size(1)
        memory_transform, context, attention_weights = self._attention_init(encoded_input, batch_size, max_length)
        h_att, c_att, h_gen, c_gen = self._decoder_init(batch_size)      
        
        # prepare some inference or train specific variables (teacher forcing, max. predicted length)
        if target is None:
            assert batch_size == 1, (f'Batch size must be 1 during inference, given {batch_size}')
            max_frames = self._max_frames
            frame = torch.zeros(1, self._output_dim)   
            frame = self._prenet(frame)   
        else:
            max_frames = target.size(2)
            target = self._target_init(target, batch_size)   
        
        # tensors for storing output
        spectrogram = torch.zeros(batch_size, max_frames, self._output_dim)
        alignments = torch.zeros(batch_size, max_frames, max_length)
        stop_tokens = torch.zeros(batch_size, max_frames, 1)
        
        # decoding loop
        for i in range(max_frames):
            prev_frame = frame if target is None else target[:,i]

            # run decoder attention and RNNs
            attention_input = torch.cat((prev_frame, context), dim=1)
            h_att, c_att = self._attention_lstm(attention_input, h_att, c_att)
            context, weights = self._attention(h_att, encoded_input, memory_transform, attention_weights, mask)
            attention_weights = attention_weights + weights.squeeze(1)
            generator_input = torch.cat((h_att, context), dim=1)
            h_gen, c_gen = self._generator_lstm(generator_input, h_gen, c_gen)
            
            # predict frame and stop token
            proto_output = torch.cat((h_gen, context), dim=1)
            frame = self._frame_prediction(proto_output)
            stop_logits = self._stop_prediction(proto_output)
            
            # store outputs
            spectrogram[:,i] = frame
            alignments[:,i] = attention_weights
            stop_tokens[:,i] = stop_logits
            
            # stop decoding if predicted (just during inference)
            if target is None and torch.sigmoid(stop_logits).ge(0.5):
                return spectrogram[:,:i+1], stop_tokens[:,:i+1].squeeze(2), alignments[:,:i+1]
        
        return spectrogram, stop_tokens.squeeze(2), alignments

    def forward(self, encoded_input, mask, target):
        return self._decode(encoded_input, mask, target)

    def inference(self, encoded_input):
        spectrogram, _, alignment = self._decode(encoded_input.unsqueeze(0))
        return spectrogram.squeeze(0), alignment.squeeze(0)
     

class Tacotron(torch.nn.Module):
    """
    Tacotron 2:
        characters as learned embedding
        encoder, attention, decoder which predicts frames of mel spectrogram
        the predicted mel spectrogram is passed through post-net which
          predicts a residual to add to the prediction
        minimize MSE from before and after the post-net to aid convergence

        TODO: Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        other_symbols = 3 # PAD, EOS, UNK
        self._embedding = Embedding(
                            hp.symbols_count() + other_symbols, 
                            hp.embedding_dimension, 
                            padding_idx=0
                        )
        self._encoder = Encoder(
                            hp.embedding_dimension, 
                            hp.encoder_dimension, 
                            hp.encoder_blocks, 
                            hp.encoder_kernel_size, 
                            hp.dropout
                        )
        self._prenet = Prenet(
                            hp.num_mels, 
                            hp.prenet_dimension, 
                            hp.prenet_layers, 
                            hp.dropout
                        )
        self._attention = Attention(
                            hp.attention_dimension, 
                            hp.decoder_dimension, 
                            hp.encoder_dimension, 
                            hp.attention_kernel_size,
                            hp.attention_location_dimension
                        )
        self._decoder = Decoder(
                            hp.num_mels, 
                            hp.decoder_dimension, 
                            self._attention, 
                            hp.encoder_dimension, 
                            self._prenet, 
                            hp.prenet_dimension,
                            hp.max_output_length
                        )
        self._postnet = Postnet(
                            hp.num_mels, 
                            hp.postnet_dimension,
                            hp.postnet_blocks, 
                            hp.postnet_kernel_size, 
                            hp.dropout
                        )

    def forward(self, text, text_length, spectrograms=None):  
        embedded = self._embedding(text)
        encoded = self._encoder(embedded, text_length)
        encoded_mask = lengths_to_mask(text_length)
        decoded = self._decoder(encoded, encoded_mask, spectrograms)
        prediction, stop_token, alignment = decoded
        prediction = prediction.transpose(1,2)
        residual = self._postnet(prediction)
        prediction += residual
        return prediction, residual, stop_token, alignment

    
class TacotronLoss(torch.nn.Module):
    """
    Tacotron 2 loss function:
        minimize the summed MSE from before and after the post-net to aid convergence

        TODO: Guided Attention Loss
    """

    def __init__(self):
        super(TacotronLoss, self).__init__()

    def forward(self, predicted_spectrogram, predicted_residual, predicted_stop_token, target_spectrogram, target_stop_token, target_lengths=None):
        if target_lengths is None: target_mask = torch.torch.ones(target_spectrogram.shape()[0,2])
        else: target_mask = lengths_to_mask(target_lengths).type(torch.FloatTensor)
        loss = F.binary_cross_entropy_with_logits(predicted_stop_token * target_mask, target_stop_token)
        target_mask = target_mask.unsqueeze(1)
        loss += F.mse_loss(predicted_spectrogram * target_mask, target_spectrogram)
        loss += F.mse_loss(predicted_residual * target_mask, target_spectrogram) 
        return loss