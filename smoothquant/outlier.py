import torch
from smoothquant.fake_outlier import OutlierLinear
from collections import Counter
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

def outlier_fix_model(model, outlier_dict:dict, upper_bound:dict = {},  weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True,):

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            fc1_name = "model." + name + ".fc1"
            fc2_name = "model." + name + ".fc2"
            fc1_upper = None if fc1_name not in upper_bound else upper_bound[fc1_name]
            fc2_upper = None if fc2_name not in upper_bound else upper_bound[fc2_name]
            m.fc1 = OutlierLinear.from_float(m.fc1, outlier_dict=outlier_dict[fc1_name], upper_bound=fc1_upper, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = OutlierLinear.from_float(m.fc2, outlier_dict=outlier_dict[fc2_name], upper_bound=fc2_upper, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            q_name = "model." + name + ".q_proj"
            k_name = "model." + name + ".k_proj"
            v_name = "model." + name + ".v_proj"
            o_name = "model." + name + ".out_proj"

            q_upper = None if q_name not in upper_bound else upper_bound[q_name]
            k_upper = None if k_name not in upper_bound else upper_bound[k_name]
            v_upper = None if v_name not in upper_bound else upper_bound[v_name]
            o_upper = None if o_name not in upper_bound else upper_bound[o_name]

            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = OutlierLinear.from_float(
                m.q_proj, outlier_dict=outlier_dict[q_name], upper_bound=q_upper, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = OutlierLinear.from_float(
                m.k_proj, outlier_dict=outlier_dict[k_name], upper_bound=k_upper, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = OutlierLinear.from_float(
                m.v_proj, outlier_dict=outlier_dict[v_name], upper_bound=v_upper, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = OutlierLinear.from_float(m.out_proj, outlier_dict=outlier_dict[o_name], upper_bound=o_upper, weight_quant=weight_quant, act_quant=act_quant)
    return model
