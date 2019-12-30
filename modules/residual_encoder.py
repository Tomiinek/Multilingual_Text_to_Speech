import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LSTM

from modules.layers import ConvBlock


class ResidualEncoder(torch.nn.Module):
    """
    Encoder:
        stack of 2 conv. layers 3 × 1 with BN and ReLU, 512 filters
        output is passed into two Bi-LSTM layers with 256 cells at each direction

    Arguments:
        input_dim -- size of the input (supposed number of mels)
        hidden_dim -- number of channels of the convolutional blocks
        latent_dim -- size of the latent vector
        output_dim -- number of channels of the last Bi-LSTM
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks, kernel_size, dropout):
        super(ResidualEncoder, self).__init__()
        assert num_blocks > 0, ('There must be at least one convolutional block in the latent encoder.')
        assert hidden_dim % 2 == 0, ('Bidirectional LSTM output dimension must be divisible by 2.')
        self._latent_dim = latent_dim
        self._convs = Sequential(
            ConvBlock(input_dim, hidden_dim, kernel_size, dropout, 'relu'),
            *[ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout, 'relu') for _ in range(num_blocks - 1)]
        )
        self._lstm = LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self._mean = Linear(hidden_dim, latent_dim)
        self._log_variance = Linear(hidden_dim, latent_dim)

    def forward(self, x, x_lenghts):

        x = self._convs(x)
        x = x.transpose(1, 2)
        ml = x.size(1)
        sorted_x_lenghts, sorted_idxs = torch.sort(x_lenghts, descending=True)
        x = x[sorted_idxs]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts[sorted_idxs], batch_first=True)
        self._lstm.flatten_parameters()
        x, _ = self._lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=ml) 
        x[sorted_idxs] = x
        x = torch.mean(x, dim=1)
        
        mean = self._mean(x)
        log_variance = self._log_variance(x)
        std = torch.exp(log_variance / 2)
        z = mean + torch.randn_like(std) * std

        return z, mean, std

    def inference(self, batch_size):
        return torch.zeros(batch_size, self._latent_dim)

    @staticmethod
    def _kl_divergence(a_mean, a_sd, b_mean, b_sd):
        """Method for computing KL divergence of two normal distributions."""
        a_sd_squared, b_sd_squared = a_sd ** 2, b_sd ** 2
        ratio = a_sd_squared / b_sd_squared
        return (a_mean - b_mean) ** 2 / (2 * b_sd_squared) + (ratio - torch.log(ratio) - 1) / 2

    @staticmethod
    def loss(mean, std, predicted=None, target=None):
        return torch.mean(ResidualEncoder._kl_divergence(mean, std, 0.0, 1.0))
        # FIXME: weight the losses
        # FIXME: BCE as reconstruction loss?