import torch
from einops import einsum
from statistics import variance
from einops import einsum
from torch.nn.init import trunc_normal_
import torch.nn as nn

class Transgender():
    pass

# class Linear():

#     def __init__(self, in_features, out_features, device=None, dtype=None):
#         super().__init__(...)

#     def forward(self, x):
#         pass
        
# def variance_and_std(in_features, out_features):
    
#     variance = 2/(in_features+out_features)
#     std = variance**(1/2)

#     return variance,std

class Transgender(nn.Module):
    def __init__(self, data = 2):
        self.data = data


def variance_and_std(in_features, out_features):
    
    variance = 2/(in_features+out_features)
    std = variance**(1/2)

    return variance,std
    
class Linear(nn.Module):
    def __init__(self, in_features=None, out_features=None, device=None, dtype=None):

        super().__init__()
        self.infeatures = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.ones((self.out_features, self.infeatures), dtype=dtype))
        variance, std = variance_and_std(self.infeatures, self.out_features)
        self.weights = trunc_normal_(self.weights, mean=0, std = std, a = -3*std, b=3*std)


    def forward(self, x : torch.Tensor):
        
        output = einsum(x, self.weights, "... d_in, d_in d_out-> ... d_out")

        return output