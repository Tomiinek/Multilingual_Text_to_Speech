import torch
from torch.nn import functional as F
from torch.nn import Sequential, ModuleList, Linear, ReLU, Sigmoid, Tanh, Dropout, LSTM, Embedding

from modules.layers import ZoneoutLSTMCell, DropoutLSTMCell, ConvBlock
from modules.attention import LocationSensitiveAttention, ForwardAttention, ForwardAttentionWithTransition
from modules.cbhg import PostnetCBHG
from params.params import Params as hp


def lengths_to_mask(lengths, max_length=None):
    ml = torch.max(lengths) if max_length is None else max_length
    return torch.arange(ml, device=lengths.device)[None, :] < lengths[:, None]


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
        ml = x.size(1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts, batch_first=True)
        self._lstm.flatten_parameters()
        x, _ = self._lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=ml) 
        return x


class Prenet(torch.nn.Module):
    """
    Prenet:
        stack of 2 linear layers with dropout which is enabled even during inference (output variation)
        should act as bottleneck for attention

    Keyword arguments:
        input_dim -- size of the input (supposed the number of frame mels)
        output_dim -- size of the output
        num_layers -- number of the linear layers (at least one)
        dropout -- dropout rate to be aplied after each layer (even during inference)
    """

    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(Prenet, self).__init__()
        assert num_layers > 0, ('There must be at least one layer in the pre-net.')
        self._dropout_rate = dropout
        self._activation = ReLU()
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
        assert num_blocks > 1, ('There must be at least two convolutional blocks in the post-net.')
        self._convs = Sequential(
            ConvBlock(input_dimension, postnet_dimension, kernel_size, dropout, 'tanh'),
            *[ConvBlock(postnet_dimension, postnet_dimension, kernel_size, dropout, 'tanh')] * (num_blocks - 2),
            ConvBlock(postnet_dimension, input_dimension, kernel_size, dropout, 'identity')
        )

    def forward(self, x, x_lengths):
        residual = x
        x = self._convs(x)
        x += residual   
        return x


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
        generator_rnn -- instance of generator RNN 
        attention_rnn -- instance of attention RNN
        context_dim -- size of the context vector produced by the given attention
        prenet -- instance of the pre-net module
        prenet_dim -- output dimension of the pre-net
        max_length -- maximal length of the input sequence
        max_frames -- maximal number of the predicted frames
    """

    def __init__(self, output_dim, decoder_dim, attention, generator_rnn, attention_rnn, context_dim, prenet, prenet_dim, max_frames):
        super(Decoder, self).__init__()
        self._prenet = prenet
        self._attention = attention
        self._output_dim = output_dim
        self._decoder_dim = decoder_dim
        self._max_frames = max_frames
        self._attention_lstm = attention_rnn
        self._generator_lstm = generator_rnn
        self._frame_prediction = Linear(context_dim + decoder_dim, output_dim)
        self._stop_prediction = Linear(context_dim + decoder_dim, 1)

    def _target_init(self, target, batch_size):
        """Prepend target spectrogram with a zero frame and pass it through pre-net."""
        # the F.pad function has some issues: https://github.com/pytorch/pytorch/issues/13058
        first_frame = torch.zeros(batch_size, self._output_dim, device=target.device).unsqueeze(1)
        target = target.transpose(1, 2) # [B, F, N_MEL]
        target = torch.cat((first_frame, target), dim=1)
        target = self._prenet(target)
        return target

    def _decoder_init(self, batch_size, device):
        """Initialize hidden and cell state of the deocder's RNNs."""
        h_att = torch.zeros(batch_size, self._decoder_dim, device=device)
        c_att = torch.zeros(batch_size, self._decoder_dim, device=device)
        h_gen = torch.zeros(batch_size, self._decoder_dim, device=device)
        c_gen = torch.zeros(batch_size, self._decoder_dim, device=device)
        return h_att, c_att, h_gen, c_gen

    def _decode(self, encoded_input, mask, target, teacher_forcing_ratio):
        """Perform decoding of the encoded input sequence."""
        
        # attention and decoder states initialization
        batch_size = encoded_input.size(0)
        max_length = encoded_input.size(1)
        inference = target is None
        max_frames = self._max_frames if inference else target.size(2) 
        input_device = encoded_input.device
        context = self._attention.reset(encoded_input, batch_size, max_length, input_device)
        h_att, c_att, h_gen, c_gen = self._decoder_init(batch_size, input_device)      
        
        # prepare some inference or train specific variables (teacher forcing, max. predicted length)
        frame = torch.zeros(batch_size, self._output_dim, device=input_device)
        target = self._target_init(target, batch_size)   
        if not inference:
            teacher = torch.rand([max_frames], device=input_device) > (1 - teacher_forcing_ratio)
        
        # tensors for storing output
        spectrogram = torch.zeros(batch_size, max_frames, self._output_dim, device=input_device)
        alignments = torch.zeros(batch_size, max_frames, max_length, device=input_device)
        stop_tokens = torch.zeros(batch_size, max_frames, 1, device=input_device)
        
        # decoding loop
        for i in range(max_frames):
            prev_frame = self._prenet(frame) if inference or not teacher[i] else target[:,i]

            # run decoder attention and RNNs
            attention_input = torch.cat((prev_frame, context), dim=1)
            h_att, c_att = self._attention_lstm(attention_input, h_att, c_att)
            context, weights = self._attention(h_att, encoded_input, mask, prev_frame)
            generator_input = torch.cat((h_att, context), dim=1)
            h_gen, c_gen = self._generator_lstm(generator_input, h_gen, c_gen)
            
            # predict frame and stop token
            proto_output = torch.cat((h_gen, context), dim=1)
            frame = self._frame_prediction(proto_output)
            stop_logits = self._stop_prediction(proto_output)
            
            # store outputs
            spectrogram[:,i] = frame
            alignments[:,i] = weights
            stop_tokens[:,i] = stop_logits
            
            # stop decoding if predicted (just during inference)
            if inference and torch.sigmoid(stop_logits).ge(0.5):
                return spectrogram[:,:i+1], stop_tokens[:,:i+1].squeeze(2), alignments[:,:i+1]
        
        return spectrogram, stop_tokens.squeeze(2), alignments

    def forward(self, encoded_input, encoded_lenghts, target, teacher_forcing_ratio):
        ml = encoded_input.size(1)
        mask = lengths_to_mask(encoded_lenghts, max_length=ml)
        return self._decode(encoded_input, mask, target, teacher_forcing_ratio)

    def inference(self, encoded_input):
        mask = lengths_to_mask([encoded_input.size(1)])
        spectrogram, _, _ = self._decode(encoded_input.unsqueeze(0), mask, None, 0.0)
        return spectrogram.squeeze(0)
     

class Tacotron(torch.nn.Module):
    """
    Tacotron 2:
        characters as learned embedding
        encoder, attention, decoder which predicts frames of mel spectrogram
        the predicted mel spectrogram is passed through post-net which
          predicts a residual to add to the prediction
        minimize MSE from before and after the post-net to aid convergence
    """

    def __init__(self):
        super(Tacotron, self).__init__()
        other_symbols = 3 # PAD, EOS, UNK
        self._embedding = Embedding(
                            hp.symbols_count() + other_symbols, 
                            hp.embedding_dimension, 
                            padding_idx=0
                        )
        torch.nn.init.xavier_uniform_(self._embedding.weight)
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
        self._attention = self._get_attention(hp.attention_type)
        gen_cell_dimension = hp.encoder_dimension + hp.decoder_dimension
        att_cell_dimension = hp.encoder_dimension + hp.prenet_dimension
        if hp.decoder_regularization == 'zoneout':
            generator_rnn = ZoneoutLSTMCell(gen_cell_dimension, hp.decoder_dimension, hp.zoneout_hidden, hp.zoneout_cell) 
            attention_rnn = ZoneoutLSTMCell(att_cell_dimension, hp.decoder_dimension, hp.zoneout_hidden, hp.zoneout_cell)
        else:
            generator_rnn = DropoutLSTMCell(gen_cell_dimension, hp.decoder_dimension, hp.dropout_hidden) 
            attention_rnn = DropoutLSTMCell(att_cell_dimension, hp.decoder_dimension, hp.dropout_hidden)
        self._decoder = Decoder(
                            hp.num_mels, 
                            hp.decoder_dimension, 
                            self._attention, 
                            generator_rnn,
                            attention_rnn, 
                            hp.encoder_dimension,
                            self._prenet, 
                            hp.prenet_dimension,
                            hp.max_output_length
                        )        
        if hp.predict_linear:
            self._postnet = PostnetCBHG(
                            hp.num_mels, 
                            hp.num_fft // 2 + 1, 
                            hp.cbhg_bank_kernels, 
                            hp.cbhg_bank_dimension, 
                            hp.cbhg_projection_dimension, 
                            hp.cbhg_projection_kernel_size,
                            hp.cbhg_highway_dimension, 
                            hp.cbhg_rnn_dim,
                            hp.cbhg_dropout
                        )
        else:
            self._postnet = Postnet(
                            hp.num_mels, 
                            hp.postnet_dimension,
                            hp.postnet_blocks, 
                            hp.postnet_kernel_size, 
                            hp.dropout
                        )
            
    def _get_attention(self, name):
        args = (hp.attention_dimension,
                hp.decoder_dimension, 
                hp.encoder_dimension)
        if name == "location_sensitive":
            return LocationSensitiveAttention(
                hp.attention_kernel_size,
                hp.attention_location_dimension,
                False,
                *args
            )
        elif name == "forward":
            return ForwardAttention(*args)
        elif name == "forward_transition_agent":
            return ForwardAttentionWithTransition(
                hp.prenet_dimension,
                *args
            )

    def forward(self, text, text_length, mel_target, target_length, speakers, teacher_forcing_ratio=0.0):  
        embedded = self._embedding(text)
        encoded = self._encoder(embedded, text_length)
        decoded = self._decoder(encoded, text_length, mel_target, teacher_forcing_ratio)
        prediction, stop_token, alignment = decoded
        pre_prediction = prediction.transpose(1,2)
        post_prediction = self._postnet(pre_prediction, target_length)

        # mask output paddings
        target_mask = lengths_to_mask(target_length, mel_target.size(2))
        # TODO: this should be somehow unmasked for few following frames :/
        stop_token.masked_fill_(~target_mask, 1000)
        target_mask = target_mask.unsqueeze(1).float()
        pre_prediction = pre_prediction * target_mask
        post_prediction = post_prediction * target_mask

        return post_prediction, pre_prediction, stop_token, alignment

    def inference(self, text, speaker_embedding=None):
        embedded = self._embedding(text)
        encoded = self._encoder(embedded, [encoded.size(1)])
        prediction = self._decoder.inference(encoded)
        prediction = prediction.transpose(1,2)
        post_prediction = self._postnet(prediction)
        return post_prediction


class TacotronLoss(torch.nn.Module):
    """Wrapper around loss functions.
    
    - L2 or L1 of the prediction before and after the postnet.
    - Cross entropy of the stop tokens (non masked)
    - Guided attention loss:
        prompt the attention matrix to be nearly diagonal, this is how people usualy read text
        introduced by 'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
    """

    def __init__(self, guided_att_epochs, guided_att_variance, guided_att_gamma):
        super(TacotronLoss, self).__init__()
        self._g = guided_att_variance
        self._gamma = guided_att_gamma
        self._g_epochs = guided_att_epochs

    def update_states(self):
        self._g *= self._gamma
        self._g_epochs = max(0, self._g_epochs - 1)

    def _guided_attention(self, alignments, input_lengths, target_lengths):
        if not hp.guided_attention_loss or self._g_epochs == 0: return 0
        input_device = alignments.device
        weights = torch.zeros_like(alignments)
        for i, (f, l) in enumerate(zip(target_lengths, input_lengths)):
            grid_f, grid_l = torch.meshgrid(torch.arange(f, dtype=torch.float, device=input_device), torch.arange(l, dtype=torch.float, device=input_device))
            weights[i, :f, :l] = 1 - torch.exp(-(grid_l/l - grid_f/f)**2 / (2 * self._g ** 2)) 
        loss = torch.sum(weights * alignments, dim=(1,2))
        loss = torch.mean(loss / target_lengths.float())
        return loss

    def forward(self, source_length, pre_prediction, pre_target, post_prediction, post_target, target_length, stop, target_stop, alignment):
        pre_target.requires_grad = False
        post_target.requires_grad = False
        target_stop.requires_grad = False
        
        # F.l1_loss

        losses = {
            'mel_pre' : 2 * F.mse_loss(pre_prediction, pre_target),
            'mel_pos' : F.l1_loss(post_prediction, post_target),
            'stop_token' : F.binary_cross_entropy_with_logits(stop, target_stop) / hp.num_mels,
            'guided_att' : self._guided_attention(alignment, source_length, target_length)
        }

        return sum(losses.values()), losses


    
