import torch
from torch.nn import functional as F
from torch.nn import Dropout, Sequential, Linear, Softmax
from utils import lengths_to_mask


class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l, c):
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return ctx.l * grad_output.neg(), None, None


class ReversalClassifier(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, gradient_clipping_bounds, scale_factor=1.0):
        super(ReversalClassifier, self).__init__()
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds
        self._output_dim = output_dim
        self._classifier = Sequential(
            Linear(input_dim, hidden_dim),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x):  
        x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        x = self._classifier(x)
        return x

    @staticmethod
    def loss(input_lengths, speakers, prediction):
        ignore_index = -100
        ml = torch.max(input_lengths)
        input_mask = torch.arange(ml, device=input_lengths.device)[None, :] < input_lengths[:, None]
        target = speakers.repeat(ml, 1).transpose(0,1)
        target[~input_mask] = ignore_index
        return F.cross_entropy(prediction.transpose(1,2), target, ignore_index=ignore_index)


#class CosineSimilarityClassifier(torch.nn.Module):
#
#    Cosine similarity-based adversarial process
#
#    def __init__(self, input_dim, hidden_dim, output_dim, gradient_clipping_bounds):
#        super(CosineSimilarityClassifier, self).__init__()
#        # self._clipping = gradient_clipping_bounds
#        self._classifier = Linear(input_dim, output_dim)
#
#    def forward(self, x):
#        return self._classifier(x)
#
#    @staticmethod
#    def loss(self, input_lengths, languages, prediction, eps = 1e-8):
#        l = ReversalClassifier.loss(input_lengths, languages, prediction)
#
#        l += cosine_loss
#        return l
#
#        w = self._classifier.weight
#
#        dot = w @ w.t()
#        norm = torch.norm(E, 2, 1)
#        x = torch.div(dot, norm)
#        x = torch.div(x, torch.unsqueeze(norm, 0))
#        return x
#
#        E = torch.randn(20000, 100)
#        embeddings_to_cosine_similarity_matrix(E)
#
#        ignore_index = -100
#        input_mask = lengths_to_mask(input_lengths)
#        target = torch.zeros_like(input_mask, dtype=torch.int64)     
#        for l in range(self._output_dim):
#            language_mask = (languages == l)
#            target[language_mask] = l
#        target[~input_mask] = ignore_index
#        return F.cross_entropy(prediction.transpose(1,2), target, ignore_index=ignore_index)