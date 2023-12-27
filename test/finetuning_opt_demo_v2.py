import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from peft.tuners.lora import LoraLayer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_lora_quant_v2 import W8A8Linear, OutlierLinear, SimpleClipLinear
from smoothquant.model import build_lora_model
from datasets import load_dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-6.7b', help='model name')
    parser.add_argument('--file-name', type=str,
                        default='opt-6.7b.pt', help='model name')
    # parser.add_argument('--output-path', type=str, default='outlier_idx/opt-1.3b.pt',
    #                     help='where to save the act scales')
    parser.add_argument('--dataset', type=str, default='lambada', help='location of the calibration dataset, we use the validation set of the oasst1 validation dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--strategy', type=str,
                        default='IQR', help='outlier detection strategy')
    parser.add_argument("--full", action='store_true', help='if test full precision model')
    parser.add_argument("--w8a8", action='store_true', help='if test naive w8a8 precision model')
    parser.add_argument("--w8", action='store_true', help='if test naive w8a8 precision model')
    parser.add_argument("--smooth", action='store_true', help='if test smooth w8a8 precision model')
    parser.add_argument("--clip_lora", action='store_true', help='if test clip+lora w8a8 precision model')
    parser.add_argument("--lora_int8", action='store_true', help='if test lora-int8 w8a8 precision model')
    parser.add_argument("--simple_clip", action='store_true', help='if test simple_clip w8a8 precision model')
    parser.add_argument("--simple_clip_lora", action='store_true', help='if test simple_clip w8a8 precision model')
    parser.add_argument("--train", action='store_true', help='if train model before testing')
    parser.add_argument("--test", action='store_true', help='if test model before training')

    args = parser.parse_args()
    return args


def linear_quant(m, quantize_bmm_input, outlier_dict={}, clip=False, B_grad=True, outlier=True):
    if clip:
        if outlier:
            return OutlierLinear.from_float(m, outlier_dict, quantize_output=quantize_bmm_input, B_grad=B_grad)
        else:
            return SimpleClipLinear.from_float(m, outlier_dict, quantize_output=quantize_bmm_input, B_grad=B_grad)
    else:
        return W8A8Linear.from_float(m, quantize_output=quantize_bmm_input)

def lora_linear_quant(m, quantize_bmm_input=False, *args, **kwargs):
    if isinstance(m, LoraLayer):
        m.base_layer = linear_quant(m.base_layer, quantize_bmm_input, *args, **kwargs)
        return m
    elif isinstance(m, nn.Linear):
        return linear_quant(m, quantize_bmm_input, *args, **kwargs)
    else:
        raise Exception("Error layer !!! ")

def quantize_opt_model(model, quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = lora_linear_quant(m.fc1)
            m.fc2 = lora_linear_quant(m.fc2)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = lora_linear_quant(m.q_proj, quantize_bmm_input)
            m.k_proj = lora_linear_quant(m.k_proj, quantize_bmm_input)
            m.v_proj = lora_linear_quant(m.v_proj, quantize_bmm_input)
            m.out_proj = lora_linear_quant(m.out_proj, quantize_bmm_input)
    return model


def quantize_outlier_opt_model(model, outlier_dict, quantize_bmm_input=True, B_grad=True, *args, **kwargs):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            fc1_name =  "model." + name + ".fc1" if B_grad else name + ".fc1"
            fc2_name = "model." + name + ".fc2" if B_grad else name + ".fc2"
            m.fc1 = lora_linear_quant(m.fc1, outlier_dict=outlier_dict[fc1_name], clip=True, B_grad=B_grad, *args, **kwargs)
            m.fc2 = lora_linear_quant(m.fc2, outlier_dict=outlier_dict[fc2_name], clip=True, B_grad=B_grad, *args, **kwargs)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            q_name = "model." + name + ".q_proj" if B_grad else name + ".q_proj"
            k_name = "model." + name + ".k_proj" if B_grad else name + ".k_proj"
            v_name = "model." + name + ".v_proj" if B_grad else name + ".v_proj"
            o_name = "model." + name + ".out_proj" if B_grad else name + ".out_proj"
            m.q_proj = lora_linear_quant(m.q_proj, quantize_bmm_input, outlier_dict=outlier_dict[q_name], clip = True, B_grad=B_grad, *args, **kwargs)
            m.k_proj = lora_linear_quant(m.k_proj, quantize_bmm_input, outlier_dict=outlier_dict[k_name], clip = True, B_grad=B_grad, *args, **kwargs)
            m.v_proj = lora_linear_quant(m.v_proj, quantize_bmm_input, outlier_dict=outlier_dict[v_name], clip = True, B_grad=B_grad, *args, **kwargs)
            m.out_proj = lora_linear_quant(m.out_proj, quantize_bmm_input, outlier_dict=outlier_dict[o_name], clip = True, B_grad=B_grad, *args, **kwargs)
    return model

class LambdaOPTTrainer:
    def __init__(self, model, train_dataset, test_dataset, tokenizer, device, batch_size=64, epochs=10):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model = model
        # tokenize the dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.trainer = transformers.Trainer(
                model=model,
                train_dataset = self.train_dataset,
                eval_dataset = self.test_dataset,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size = self.batch_size,
                    gradient_accumulation_steps = 4,
                    warmup_steps=100,
                    # max_steps=200,
                    num_train_epochs = epochs,
                    learning_rate=1e-3,
                    fp16=True,
                    # logging_steps=10,
                    output_dir='outputs'
                ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

    def train(self):
        self.trainer.train()

    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.test_dataset):
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
    batch_size = args.batch_size
    epochs = args.epochs

    tokenizer = GPT2Tokenizer.from_pretrained(model_name,load_in_8bit=True,)

    if args.dataset == "lambada":
        train_dataset = load_dataset('lambada', split='validation')
        test_dataset = load_dataset('lambada', split='test')

        def tokenize_function(examples):
            example = tokenizer(examples['text'])
            return example

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids'])

        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids'])
    else:
        raise Exception("TODO")

    if args.full:
        model = build_lora_model(model_name)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

        if args.test:
            acc_fp32 = trainer.evaluate(model)
            print(f'Original model (fp32) before fine tuning accuracy: {acc_fp32}')
        if args.train:
            trainer.train()
            acc_fp32 = trainer.evaluate(model)
            print(f'Original model (fp32) after fine tuning accuracy: {acc_fp32}')

    if args.w8a8:
        model = build_lora_model(model_name)
        model = quantize_opt_model(model, )
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

        if args.test:
            acc_w8a8 = trainer.evaluate(model)
            print(f'Naive W8A8 quantized model before fine tuning accuracy: {acc_w8a8}')

        if args.train:
            trainer.train()
            acc_w8a8 = trainer.evaluate(model)
            print(f'Naive W8A8 quantized model after fine tuning accuracy: {acc_w8a8}')

    # model = build_lora_model(model_name)
    # model = quantize_lora_model(model, act_quant="None", weight_quant=weight_quant)
    # trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
    #
    # acc_w8 = trainer.evaluate(model)
    # print(f'w8 model before fine tuning accuracy: {acc_w8}')
    #
    # trainer.train()
    #
    # acc_w8 = trainer.evaluate(model)
    # print(f'w8 model after fine tuning accuracy: {acc_w8}')

    if args.smooth:
        model = build_lora_model(model_name)
        act_scales = torch.load('act_scales/' + args.file_name)  # it is generated before test
        smooth_lm(model, act_scales, 0.5)
        model = quantize_opt_model(model,)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_smooth = trainer.evaluate(model)
            print(f'SmoothQuant W8A8 quantized model before fine tuning accuracy: {acc_smooth}')

        if args.train:
            trainer.train()
            acc_smooth = trainer.evaluate(model)
            print(f'SmoothQuant W8A8 quantized model after fine tuning accuracy: {acc_smooth}')

    if args.clip_lora:
        filename = f"{args.strategy}_{args.file_name}"
        model = build_lora_model(model_name)
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        model = quantize_outlier_opt_model(model, outlier_dict=outlier_indices, B_grad=False)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate(model)
            print(f'clip_lora {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate(model)
            print(f'clip_lora {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')

    if args.lora_int8:
        filename = f"{args.strategy}_{args.file_name}"
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='auto')
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        model = quantize_outlier_opt_model(model, outlier_dict=outlier_indices, B_grad=True)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate(model)
            print(f'lora_int8 {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate(model)
            print(f'lora_int8 {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')

    if args.simple_clip:
        filename = f"{args.strategy}_{args.file_name}"
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='auto')
        outlier_indices = torch.load('outlier_idx/' + filename)  # it is generated before test
        model = quantize_outlier_opt_model(model, outlier_dict=outlier_indices, outlier=True)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate(model)
            print(f'simple_clip_lora {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate(model)
            print(f'simple_clip_lora {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')

    if args.simple_clip_lora:
        filename = f"{args.strategy}_{args.file_name}"
        build_lora_model(model_name)
        outlier_indices = torch.load('outlier_idx/' + filename)  # it is generated before test
        model = quantize_outlier_opt_model(model, outlier_dict=outlier_indices,  B_grad=False, outlier=True)
        trainer = LambdaOPTTrainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate(model)
            print(f'simple_clip_lora {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate(model)
            print(f'simple_clip_lora {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')
