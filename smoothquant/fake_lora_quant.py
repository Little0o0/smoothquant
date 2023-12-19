import torch
from torch import nn
from functools import partial
from torch.autograd import Function
import torch.nn.functional as F



def per_channel_quantization_absmax(t, n_bits=8):
    # t: ( out_channel * in_channel ) or ( batch * token * in_channel )
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits-1)-1
    scales = scales.clamp(min=1e-5).div(q_max)
    t = t.div(scales).round().mul(scales)
    return t

def per_tensor_quantization_absmax(t, n_bits=8):
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    t = t.div(scales).round().mul(scales)
    return t

# TODO
# add per_tensor_clipping etc


Quant_Functions = {
    "per_channel_absmax": per_channel_quantization_absmax,
    "per_tensor_absmax": per_tensor_quantization_absmax,

}


class QuantLinearFunction(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias,
                flag_quant = True,
                act_quant="per_tensor_absmax",
                weight_quant="per_tensor_absmax",
                indices = None):

        if not flag_quant:
            output = F.linear(input, weight)
        else:
            Qa = Quant_Functions[act_quant]
            Qw = Quant_Functions[weight_quant]
            output = F.linear(Qa(input), Qw(weight))
        if bias is not None:
            output += bias.view(1,1,-1).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias, _, _, _, _ = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias