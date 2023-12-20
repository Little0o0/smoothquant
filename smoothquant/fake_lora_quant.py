from typing import Any

import torch
from torch import nn
from functools import partial
from torch.autograd import Function
from collections import Counter

class per_channel_quantization_absmax(Function):
    @staticmethod
    def forward(ctx, t, n_bits = 8, *args, **kwargs):
        # t: ( out_channel * in_channel ) or ( batch * token * in_channel )
        scales = t.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales = scales.clamp(min=1e-5).div(q_max)
        t = t.div(scales).round().mul(scales)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class per_tensor_quantization_absmax(Function):
    @staticmethod
    def forward(ctx, t, n_bits = 8, *args, **kwargs):
        scales = t.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales = scales.clamp(min=1e-5).div(q_max)
        t = t.div(scales).round().mul(scales)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



class no_quantization(Function):
    @staticmethod
    def forward(ctx, t, *args, **kwargs):
        return t

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def clipping_activation(t, outlier_idx=None,):
    upper_bound = get_upper_bound(t, outlier_idx=outlier_idx)
    # max_range = torch.min(t.abs().max(), upper_bound.to(t.device))
    t = t.clamp(min = -upper_bound, max = upper_bound)
    return t

def get_upper_bound(t, outlier_idx=None):
    if outlier_idx is None:
        return t.abs().max()
    in_features = t.shape[-1]
    select_channels = [i for i in range(in_features) if i not in outlier_idx]
    upper_bound = t[..., select_channels].abs().max()
    return upper_bound

def clipping_activation(t, upper_bound=None):
    t = t.clamp(min=-upper_bound, max=upper_bound)
    return t


quant_func = {
    "per_channel_absmax" : per_channel_quantization_absmax,
    "per_tensor_absmax" : per_tensor_quantization_absmax,
    "None" : no_quantization
}


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant = 'per_tensor_absmax', quantize_output=False, outlier_idx:list = [],):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.outlier_idx = outlier_idx
        self.outlier_features = len(self.outlier_idx)
        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if self.outlier_features != 0:
            self.register_buffer('out_weight', torch.randn(self.out_features,
                                                          self.outlier_features,
                                                          dtype=torch.float16,
                                                          requires_grad=False))

        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        self.Qa = quant_func[act_quant].apply

        if quantize_output:
            self.Qo = self.Qa
        else:
            self.Qo = quant_func["None"].apply

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def forward(self, x):
        if self.outlier_features != 0:
            up = get_upper_bound(x, self.outlier_idx)
            up = up.to(x.device)
            x = clipping_activation(x, up)
            out_x = x[..., self.outlier_idx]
            out_x = torch.sign(out_x) * torch.maximum(torch.abs(out_x) - up, torch.tensor(0.))
            res_y = torch.functional.F.linear(out_x, self.out_weight)
        else:
            res_y = torch.tensor(0.).to(torch.float16).to(x.device)

        q_x = self.Qa(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias) + res_y
        q_y = self.Qo(y)

        return q_y

    @staticmethod
    def from_float(module, outlier_dict:Counter={}, act_quant = 'per_tensor_absmax', weight_quant ='per_tensor_absmax' , quantize_output=False):

        assert isinstance(module, torch.nn.Linear)

        if len(outlier_dict) == 0:
            max_outlier_features = module.in_features // 100
            outlier_idx = [x[0] for x in outlier_dict.most_common(max_outlier_features)]
        else:
            outlier_idx = []

        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output, outlier_idx=outlier_idx)

        Qw = quant_func[weight_quant].apply

        new_module.weight = Qw(module.weight)
        if new_module.outlier_features != 0:
            new_module.out_weight = module.weight[..., outlier_idx].clone()

        if module.bias is not None:
            new_module.bias = module.bias

        return new_module