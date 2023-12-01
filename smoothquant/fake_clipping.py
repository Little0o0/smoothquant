import torch
from torch import nn
from functools import partial


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
def clipping_activation(t, quant_step=2**-4, n_bits=8):
    q_max = 2 ** (n_bits - 1) - 1
    max_range = torch.min(t.abs().max(),torch.tensor(q_max*quant_step))
    t = t.clamp(min=-max_range, max=max_range)
    return t


@torch.no_grad()
def clipping_quantize_activation_per_token_absmax(t, clipping=True, quant_step=2**-4, n_bits=8):
    if clipping:
        t = clipping_activation(t, quant_step, n_bits)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    max_range = t.abs().max(dim=-1, keepdim=True)[0]
    scales = max_range.clamp(min=1e-5).div(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def clipping_quantize_activation_per_tensor_absmax(t, clipping=True, quant_step=2**-4, n_bits=8):
    if clipping:
        t = clipping_activation(t, quant_step, n_bits)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    max_range = t.abs().max()
    scales = max_range.clamp(min=1e-5).div(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

class ClippingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quant_step=2**-4,  act_quant='per_token', quantize_output=False, clipping=True):
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
                clipping_quantize_activation_per_token_absmax, clipping=clipping, quant_step=quant_step, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                clipping_quantize_activation_per_tensor_absmax, clipping=clipping, quant_step=quant_step, n_bits=8)
        elif act_quant == 'clipping_only':
            if clipping:
                self.act_quant_name = 'clipping_only'
                self.act_quant =  partial(
                    clipping_activation, quant_step=quant_step, n_bits=8)
            else:
                self.act_quant_name = 'None'
                self.act_quant = lambda x: x
                
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
        super(ClippingLinear, self).to(*args, **kwargs)
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
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quant_step=2**-4, quantize_output=False, clipping=True):
        assert isinstance(module, torch.nn.Linear)
        new_module = ClippingLinear(
            module.in_features, module.out_features, module.bias is not None, quant_step=quant_step, act_quant=act_quant, quantize_output=quantize_output, clipping=clipping)
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
        return f'ClippingLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'
