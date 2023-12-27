import torch
from torch import nn
from functools import partial
from torch.autograd import Function
from collections import Counter

class uniform_quant(Function):
    @staticmethod
    def forward(ctx, t, n_bits=8, *args, **kwargs):
        scales = t.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales = scales.clamp(min=1e-5).div(q_max)
        t = t.div(scales).round().mul(scales)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class no_quant(Function):
    @staticmethod
    def forward(ctx, t, *args, **kwargs):
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# quant_func = {
#     "quant": uniform_quant.apply,
#     "None": no_quant.apply
# }

def get_upper_bound(t, outlier_idx=None):
    if outlier_idx is None:
        return t.abs().max()
    in_features = t.shape[-1]
    select_channels = [i for i in range(in_features) if i not in outlier_idx]
    upper_bound = t[..., select_channels].abs().max()
    return upper_bound

class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantize_output=False, nbits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nbits = nbits
        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float32, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float32, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        self.Qa = uniform_quant.apply

        if quantize_output:
            self.Qo = self.Qa
        else:
            self.Qo = no_quant.apply

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def forward(self, x):
        q_x = self.Qa(x, self.nbits)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.Qo(y, self.nbits)

        return q_y

    @staticmethod
    def from_float(module, quantize_output=False, nbits=8):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, quantize_output=quantize_output, nbits=nbits)

        Qw = uniform_quant.apply
        new_module.weight = Qw(module.weight, nbits)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module


class OutlierLinear(nn.Module):
    def __init__(self, in_features, out_features, outlier_idx, bias=True, quantize_output=False, nbits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.outlier_idx = outlier_idx
        self.outlier_features = len(outlier_idx)
        self.nbits = nbits
        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float32, requires_grad=False))
        if self.outlier_features != 0:

            self.B = nn.parameter.Parameter(torch.zeros(self.out_features, self.outlier_features, dtype=torch.float32), requires_grad=True)

        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float32, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        self.Qa = uniform_quant.apply

        if quantize_output:
            self.Qo = self.Qa
        else:
            self.Qo = no_quant.apply

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.B = self.B.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def forward(self, x):
        if self.outlier_features != 0:
            upper = get_upper_bound(x, self.outlier_idx)
            x_clip = x.clamp(min = -upper, max = upper)
            x_outlier = x[..., self.outlier_idx] - x_clip[..., self.outlier_idx]
            res_y = torch.functional.F.linear(x_outlier, self.B)
        else:
            x_clip = x
            res_y = 0
        q_x = self.Qa(x_clip, self.nbits)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.Qo(y, self.nbits) + res_y
        return q_y

    @staticmethod
    def from_float(module, outlier_dict, quantize_output=False, nbits=8, B_grad=True):
        assert isinstance(module, torch.nn.Linear)

        if len(outlier_dict) != 0:
            max_outlier_features = module.in_features // 100
            outlier_idx = [x[0] for x in outlier_dict.most_common(max_outlier_features)]
        else:
            outlier_idx = []

        new_module = OutlierLinear(
            module.in_features, module.out_features, outlier_idx,  module.bias is not None, quantize_output=quantize_output,
            nbits=nbits)

        Qw = uniform_quant.apply

        if new_module.outlier_features != 0:
            new_module.B.data = module.weight.data[...,outlier_idx]
            new_module.B.requires_grad = B_grad

        new_module.weight = Qw(module.weight, nbits)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module