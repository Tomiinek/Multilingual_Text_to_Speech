import torch
from torch.nn import Sequential, Linear, Softmax


class GradientReversalFunction(torch.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l):
        ctx.save_for_backward(l)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        l = ctx.saved_tensors[0]
        return l * grad_output.neg(), None


class ReversalClassifier(torch.nn.Module):

    def __init__(self, input_dim, output_dim, scale_factor=1.0):
        super(ReversalClassifier, self).__init__()
        self._lambda = scale_factor
        self._layers = Sequential([
            Linear(input_dim, output_dim),
            Softmax(output_dim)
        ])

    def forward(self, x):  
        x = GradientReversalFunction.apply(x, self._lambda)
        x = self._layers(x)
        return x