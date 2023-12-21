import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
from smoothquant.fake_clipping import ClippingLinear
from smoothquant.fake_outlier import IQRClippingLinear
from smoothquant.outlier import outlier_fix_model, outlier_fix_model_v2
from datasets import load_dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--file-name', type=str,
                        default='opt-1.3b.pt', help='model name')
    # parser.add_argument('--output-path', type=str, default='outlier_idx/opt-1.3b.pt',
    #                     help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='OpenAssistant/oasst1',
                        help='location of the calibration dataset, we use the validation set of the oasst1 validation dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--strategy', type=str,
                        default='IQR_total', help='outlier detection strategy')
    parser.add_argument('--act_quant', type=str, default="per_tensor",
                        help='for clipping only [None|per_tensor|per_token|fake_clipping_quantize]')
    parser.add_argument('--weight_quant', type=str, default="per_tensor",
                        help='for clipping only [None|per_tensor|per_token|fake_clipping_quantize]')
    parser.add_argument("--full", action='store_true', help='if test full precision model')
    parser.add_argument("--w8a8", action='store_true', help='if test naive w8a8 precision model')
    parser.add_argument("--w8", action='store_true', help='if test naive w8a8 precision model')
    parser.add_argument("--smooth", action='store_true', help='if test smooth w8a8 precision model')
    parser.add_argument("--clip", action='store_true', help='if test clipping w8a8 precision model')
    parser.add_argument("--clip_v2", action='store_true', help='if test clipping w8a8 precision model')
    args = parser.parse_args()
    return args



def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


def IQRclipping_model(model, weight_quant='per_tensor',
                   act_quant='per_tensor',
                   quantize_bmm_input=True,):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = IQRClippingLinear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = IQRClippingLinear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = IQRClippingLinear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = IQRClippingLinear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = IQRClippingLinear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = IQRClippingLinear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


def clipping_model(model, weight_quant='per_tensor',
                   act_quant='per_tensor',
                   quantize_bmm_input=True,
                   quant_step=2**-4, ignore_layer=[]):
    # act_quant : "per_tensor", "per_token", "clipping_only"
    for name, m in model.model.named_modules():
        if name in ignore_layer:
            clipping = False
        else:
            clipping = True
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = ClippingLinear.from_float(m.fc1, quant_step=quant_step, weight_quant=weight_quant, act_quant=act_quant, clipping=clipping)
            m.fc2 = ClippingLinear.from_float(m.fc2, quant_step=quant_step, weight_quant=weight_quant, act_quant=act_quant, clipping=clipping)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = ClippingLinear.from_float(
                m.q_proj, quant_step=quant_step, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, clipping=clipping)
            m.k_proj = ClippingLinear.from_float(
                m.k_proj, quant_step=quant_step, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, clipping=clipping)
            m.v_proj = ClippingLinear.from_float(
                m.v_proj, quant_step=quant_step, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, clipping=clipping)
            m.out_proj = ClippingLinear.from_float(m.out_proj, quant_step=quant_step,weight_quant=weight_quant, act_quant=act_quant, clipping=clipping)
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc



if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    # filename = f"{args.strategy}_{args.file_name}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = load_dataset('lambada', split='test')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')

    if args.full:
        model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        acc_fp16 = evaluator.evaluate(model_fp16)
        print(f'Original model (fp16) accuracy: {acc_fp16}')

    if args.w8a8:
        model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        model_w8a8 = quantize_model(model_fp16)
        acc_w8a8 = evaluator.evaluate(model_w8a8)
        print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')

    if args.w8:
        model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        model_w8 = clipping_model(model_fp16, act_quant="None")
        acc_w8 = evaluator.evaluate(model_w8)
        print(f'w8 model accuracy: {acc_w8}')

    if args.smooth:
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        act_scales = torch.load('act_scales/'+ args.file_name) # it is generated before test
        smooth_lm(model, act_scales, 0.5, False)
        model_smoothquant_w8a8 = quantize_model(model)
        acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
        print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')

    if args.clip:
        filename = f"{args.strategy}_{args.file_name}"
        model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        # upper_bound = torch.load('upper_bound/'+filename)
        model_quant = outlier_fix_model(model_fp16, act_quant=args.act_quant ,outlier_dict=outlier_indices, upper_bound={})
        acc =  evaluator.evaluate(model_quant)
        print(filename + f'outlier quantized model , accuracy: {acc}')

    if args.clip_v2:
        filename = f"{args.strategy}_{args.file_name}"
        act_quant = args.act_quant + "_clip"
        weight_quant = args.weight_quant + "_clip"
        model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        # upper_bound = torch.load('upper_bound/'+filename)
        model_quant = outlier_fix_model_v2(model_fp16, act_quant=act_quant , weight_quant=weight_quant, outlier_dict=outlier_indices, upper_bound={})
        acc =  evaluator.evaluate(model_quant)
        print(filename + f'outlier quantized model , accuracy: {acc}')