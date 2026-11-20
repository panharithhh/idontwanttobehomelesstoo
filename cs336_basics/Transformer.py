import torch
from einops import einsum
from statistics import variance
from einops import einsum
from torch.nn.init import trunc_normal_
import torch.nn as nn

#   in_dim (int): The size of the input dimension
#         out_dim (int): The size of the output dimension
#         weights (Float[Tensor, "d_out d_in"]): The linear weights to use
#         in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

class Transgender:
    pass


def variance_and_std(in_features, out_features):
    var = 2/(in_features+out_features)
    std = (var)**(1/2)
    return std

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights ,device=None, dtype=None):

        super().__init__()
        self.infeatures = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(weights)
        std = variance_and_std(self.infeatures, self.out_features)
        self.weights = trunc_normal_(self.weights, mean=0, std = std, a = -3*std, b=3*std)


    def forward(self, x : torch.Tensor):
        # x is (4,12,64)
        # w is (128,64)
        output = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

        return output
