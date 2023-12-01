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
from smoothquant.outlier import outlier_fix_model
from datasets import load_dataset
from tqdm import tqdm

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
    model_name = 'facebook/opt-1.3b'
    # model_name = 'facebook/opt-6.7b'
    filename = 'opt-1.3b.pt'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')
    # model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')


    # model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    # acc_fp16 = evaluator.evaluate(model_fp16)
    # print(f'Original model (fp16) accuracy: {acc_fp16}')
    #
    # model_w8a8 = quantize_model(model_fp16)
    # # print(model_w8a8)
    # acc_w8a8 = evaluator.evaluate(model_w8a8)
    # print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')
    #
    # model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    # model_w8 = clipping_model(model_fp16, act_quant="None")
    # acc_w8 = evaluator.evaluate(model_w8)
    # print(f'w8 model accuracy: {acc_w8}')


    # quant_types = {
    #     "Clipping W8A8": "per_tensor",
    #     "Clipping W8" : "clipping_only"
    # }
    # for qt_name, quant_type in quant_types.items():
    #     model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    #     model_quant = IQRclipping_model(model_fp16 , act_quant=quant_type)
    #     acc = evaluator.evaluate(model_quant)
    #     print(f'{qt_name} quantized model , accuracy: {acc}')
    #
    # model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    # act_scales = torch.load('act_scales/'+filename) # it is generated before test
    # smooth_lm(model, act_scales, 0.5)
    # model_smoothquant_w8a8 = quantize_model(model)
    # # print(model_smoothquant_w8a8)
    # acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
    # print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')

    model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
    upper_bound = torch.load('upper_bound/'+filename)
    model_quant = outlier_fix_model(model_fp16, outlier_dict=outlier_indices, upper_bound=upper_bound)
    acc =  evaluator.evaluate(model_quant)
    print(f'outlier quantized model , accuracy: {acc}')

