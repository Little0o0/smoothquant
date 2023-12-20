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
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from smoothquant.smooth import smooth_lm
from smoothquant.fake_lora_quant import W8A8Linear
from smoothquant.fake_clipping import ClippingLinear
from smoothquant.fake_outlier import IQRClippingLinear
from smoothquant.model import build_lora_model
from smoothquant.outlier import outlier_fix_model, lora_outlier_model
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
    parser.add_argument('--dataset-path', type=str, default='OpenAssistant/oasst1',
                        help='location of the calibration dataset, we use the validation set of the oasst1 validation dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--strategy', type=str,
                        default='IQR_total', help='outlier detection strategy')
    parser.add_argument('--act_quant', type=str, default="per_tensor_absmax",
                        help='for clipping only [None|per_tensor_absmax|]')

    args = parser.parse_args()
    return args


def linear_quant(m, weight_quant, act_quant, quantize_bmm_input):
    return W8A8Linear.from_float(m, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)

def lora_linear_quant(m, weight_quant, act_quant, quantize_bmm_input=False):
    if isinstance(m, LoraLayer):
        m.base_layer = linear_quant(m.base_layer, weight_quant, act_quant, quantize_bmm_input)
        return m
    elif isinstance(m, nn.Linear):
        return linear_quant(m, weight_quant, act_quant, quantize_bmm_input)
    else:
        raise Exception("Error layer !!! ")

def quantize_lora_model(model, weight_quant='per_tensor_absmax', act_quant='per_tensor_absmax', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = lora_linear_quant(m.fc1, weight_quant, act_quant)
            m.fc2 = lora_linear_quant(m.fc2, weight_quant, act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = lora_linear_quant(m.q_proj, weight_quant, act_quant, quantize_bmm_input)
            m.k_proj = lora_linear_quant(m.k_proj, weight_quant, act_quant, quantize_bmm_input)
            m.v_proj = lora_linear_quant(m.v_proj, weight_quant, act_quant, quantize_bmm_input)
            m.out_proj = lora_linear_quant(m.out_proj, weight_quant, act_quant, quantize_bmm_input)
    return model


def IQRclipping_model(model, weight_quant='per_tensor_absmax',
                   act_quant='per_tensor_absmax',
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



class Trainer:
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
                    per_device_train_batch_size=self.batch_size,
                    # gradient_accumulation_steps=4,
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

    train_dataset = load_dataset('lambada', split='validation')
    test_dataset = load_dataset('lambada', split='test')

    def tokenize_function(examples):
        example = tokenizer(examples['text'])
        return example

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids'])

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids'])

    model = build_lora_model(model_name)
    trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

    acc_fp32 = trainer.evaluate(model)
    print(f'Original model (fp32) before fine tuning accuracy: {acc_fp32}')

    trainer.train()

    acc_fp32 = trainer.evaluate(model)
    print(f'Original model (fp32) after fine tuning accuracy: {acc_fp32}')

    model = build_lora_model(model_name)
    model = quantize_lora_model(model)
    trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

    acc_w8a8 = trainer.evaluate(model)
    print(f'Naive W8A8 quantized model before fine tuning accuracy: {acc_w8a8}')

    trainer.train()
    acc_w8a8 = trainer.evaluate(model)
    print(f'Naive W8A8 quantized model after fine tuning accuracy: {acc_w8a8}')

    # model = build_lora_model(model_name)
    # model_w8 = quantize_lora_model(model, act_quant="None")
    # trainer = Trainer(model_w8, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
    #
    # acc_w8 = trainer.evaluate(model_w8)
    # print(f'w8 model before fine tuning accuracy: {acc_w8}')
    #
    # trainer.train()
    #
    # acc_w8 = trainer.evaluate(model_w8)
    # print(f'w8 model after fine tuning accuracy: {acc_w8}')

    model = build_lora_model(model_name)
    act_scales = torch.load('act_scales/' + args.file_name)  # it is generated before test
    smooth_lm(model, act_scales, 0.5)

    model = quantize_lora_model(model)
    trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

    acc_smooth = trainer.evaluate(model)
    print(f'SmoothQuant W8A8 quantized model before fine tuning accuracy: {acc_smooth}')

    trainer.train()

    acc_smooth = trainer.evaluate(model)
    print(f'SmoothQuant W8A8 quantized model after fine tuning accuracy: {acc_smooth}')


    filename = f"{args.strategy}_{args.file_name}"
    model = build_lora_model(model_name)
    outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test

    model = lora_outlier_model(model, act_quant=args.act_quant, outlier_dict=outlier_indices,)
    trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)

    acc_lora = trainer.evaluate(model)
    print(f'LoRA {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

    trainer.train()

    acc_lora = trainer.evaluate(model)
    print(f'LoRA {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')