import logging

import torch
from torch import nn
from functools import partial
from collections import Counter

def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def clipping_activation(t, upper_bound=None, outlier_idx=None,):
    if upper_bound is None:
        upper_bound = get_upper_bound(t, outlier_idx=outlier_idx)
    max_range = torch.min(t.abs().max(), upper_bound.to(t.device))
    t = t.clamp(min=-max_range, max=max_range)
    return t

def get_upper_bound(t, outlier_idx=None):

    if outlier_idx is None:
        return t.abs().max()

    in_features = t.shape(-1)
    select_channels = [i for i in range(in_features) if i not in outlier_idx]
    # upper_bound = t[..., select_channels].abs().max()
    upper_bound = t.abs().max()
    return upper_bound
    # t = t.abs()
    # q = torch.tensor([0.25, 0.75]).to(t.device)
    # q1, q3 = torch.quantile(t.float(), q)
    # upper_bound = q3 + 1.5 * (q3 - q1)
    # lower_bound = q1 - 1.5 * (q3 - q1)
    # return torch.max(upper_bound.abs(), lower_bound.abs())

    # max_v = torch.tensor(16.).to(t.device)
    # if max_v >= t.abs().max():
    #     return t.abs().max()
    # q = torch.tensor([0.99]).to(t.device)
    # q99 = torch.quantile(t.float(), q)
    # upper_bound = torch.max(max_v, q99)
    # return torch.max(max_v, upper_bound).to(torch.float16)

    # return t.abs().max()

    # return torch.tensor(16).to(t.device).to(torch.float16)

@torch.no_grad()
def clipping_quantize_activation_per_token_absmax(t, upper_bound=None, n_bits=8, outlier_idx=None):
    t = clipping_activation(t, upper_bound=upper_bound, outlier_idx=outlier_idx)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    max_range = t.abs().max(dim=-1, keepdim=True)[0]
    scales = max_range.clamp(min=1e-5).div(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def clipping_quantize_activation_per_tensor_absmax(t, outlier_idx=None, upper_bound=None, n_bits=8):
    t = clipping_activation(t, upper_bound=upper_bound, outlier_idx=outlier_idx)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    max_range = t.abs().max()
    scales = max_range.clamp(min=1e-5).div(q_max)
    t.div_(scales).round_().mul_(scales)
    return t.to(torch.float16)

class IQRClippingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,  act_quant='per_token', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                clipping_quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                clipping_quantize_activation_per_tensor_absmax, n_bits=8)
        elif act_quant == 'clipping_only':
            self.act_quant_name = 'clipping_only'
            self.act_quant = clipping_activation
                
        elif act_quant == 'None':
            self.act_quant_name = 'None'
            self.act_quant = lambda x: x
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(IQRClippingLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = IQRClippingLinear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'IQRClippingLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'


class OutlierLinear(nn.Module):
    def __init__(self, in_features, out_features, outlier_idx:list, upper_bound=None, bias=True, act_quant='per_tensor', quantize_output=False):
        # act_quant: per_tensor only !!!

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.outlier_idx = outlier_idx
        self.outlier_features = len(outlier_idx)
        self.upper_bound = upper_bound

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

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                clipping_quantize_activation_per_token_absmax, outlier_idx=outlier_idx, upper_bound=upper_bound, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                clipping_quantize_activation_per_tensor_absmax, outlier_idx=outlier_idx,upper_bound=upper_bound, n_bits=8)
        elif act_quant == 'clipping_only':
            self.act_quant_name = 'clipping_only'
            self.act_quant =  partial(clipping_activation, outlier_idx=outlier_idx,upper_bound=upper_bound)
        elif act_quant == 'None':
            self.act_quant_name = 'None'
            self.act_quant = lambda x: x
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(OutlierLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.out_weight = self.out_weight.to(*args, **kwargs)
        self.upper_bound = self.upper_bound.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        up = self.upper_bound if self.upper_bound is not None else get_upper_bound(x)
        up = up.to(q_x.device)
        if self.outlier_features != 0:
            out_x = x[..., self.outlier_idx]
            out_x -= out_x.clamp(min=-up, max=up)
            res_y = torch.functional.F.linear(out_x, self.out_weight)
        else:
            res_y = 0
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y + res_y

    @staticmethod
    def from_float(module, outlier_dict:Counter, upper_bound:dict=None, weight_quant='per_tensor', act_quant='per_tensor', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        max_outlier_features = module.in_features // 100
        outlier_idx = [x[0] for x in outlier_dict.most_common(max_outlier_features)]
        new_module = OutlierLinear(
            module.in_features, module.out_features, outlier_idx=outlier_idx, upper_bound=upper_bound, bias = module.bias is not None, act_quant=act_quant,
            quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
            if new_module.outlier_features != 0:
                new_module.out_weight = module.weight[...,outlier_idx]
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
            if new_module.outlier_features != 0:
                new_module.out_weight = module.weight[..., outlier_idx]
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'OutlierLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'