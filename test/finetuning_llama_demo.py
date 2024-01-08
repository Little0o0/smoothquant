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
from transformers import  AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora import LoraLayer
from peft.tuners import lora
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from smoothquant.smooth import smooth_lm
from smoothquant.fake_lora_quant import W8A8Linear, SimpleClipingLinear
from smoothquant.fake_clipping import ClippingLinear
from smoothquant.fake_outlier import IQRClippingLinear
from smoothquant.model import build_lora_model, build_tokenizer
from smoothquant.outlier import lora_outlier_model, lora_outlier_model_v2
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
import evaluate
from smoothquant.datasets import make_data_module


IGNORE_INDEX = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='meta-llama/Llama-2-7b-hf', help='model name')
    parser.add_argument('--file-name', type=str,
                        default='Llama-2-7b-hf.pt', help='model name')
    # parser.add_argument('--output-path', type=str, default='outlier_idx/opt-1.3b.pt',
    #                     help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='alpaca-clean',
                        help='location of the calibration dataset, we use the validation set of the alpaca validation dataset')

    parser.add_argument('--dataset_format', type=str, default=None)

    parser.add_argument('--mmlu_dataset', type=str, default='mmlu-fs')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--source_max_len', type=int, default=1024)
    parser.add_argument('--target_max_len', type=int, default=256)


    parser.add_argument('--max_eval_samples', type=int, default=1024)
    parser.add_argument('--max_train_samples', type=int, default=4096)

    parser.add_argument('--group_by_length', type=bool, default=False)
    parser.add_argument('--train_on_source', type=bool, default=False)
    parser.add_argument('--predict_with_generate', type=bool, default=False)
    parser.add_argument('--eval_dataset_size', type=float, default=0.2)

    parser.add_argument('--strategy', type=str,
                        default='IQR', help='outlier detection strategy')

    parser.add_argument("--mmlu_split", type=str,
                        default='eval', help='eval or test')

    parser.add_argument('--act_quant', type=str, default="per_tensor",
                        help='for clipping only [None|per_channel|per_tensor]')
    parser.add_argument('--weight_quant', type=str, default="per_tensor",
                        help='for clipping only [None|per_channel|per_tensor]')
    parser.add_argument("--full", action='store_true', help='if test full precision model')
    parser.add_argument("--w8a8", action='store_true', help='if test naive w8a8 precision model')
    parser.add_argument("--smooth", action='store_true', help='if test smooth w8a8 precision model')
    parser.add_argument("--clip", action='store_true', help='if test clipping w8a8 precision model')
    parser.add_argument("--clip_v2", action='store_true', help='if test clipping w8a8 precision model')
    parser.add_argument("--simple_clip", action='store_true', help='if test clipping w8a8 precision model')
    parser.add_argument("--train", action='store_true', help='if train model before testing')
    parser.add_argument("--test", action='store_true', help='if test model before training')
    parser.add_argument('--mmlu', action='store_true', help='if evaluate with mmlu')

    args = parser.parse_args()
    return args

def linear_quant(m, weight_quant, act_quant, quantize_bmm_input):
    return W8A8Linear.from_float(m, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)

def lora_linear_quant(m, weight_quant, act_quant, quantize_bmm_input=False):
    if isinstance(m, lora.Linear):
        m.base_layer = linear_quant(m.base_layer, weight_quant, act_quant, quantize_bmm_input)
        return m
    elif isinstance(m, nn.Linear):
        return linear_quant(m, weight_quant, act_quant, quantize_bmm_input)
    else:
        raise Exception("Error layer !!! ")

def quantize_lora_model(model, weight_quant='per_tensor_absmax', act_quant='per_tensor_absmax', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = lora_linear_quant(m.gate_proj, weight_quant, act_quant)
            m.up_proj = lora_linear_quant(m.up_proj, weight_quant, act_quant)
            m.down_proj = lora_linear_quant(m.down_proj, weight_quant, act_quant)
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = lora_linear_quant(m.q_proj, weight_quant, act_quant, quantize_bmm_input)
            m.k_proj = lora_linear_quant(m.k_proj, weight_quant, act_quant, quantize_bmm_input)
            m.v_proj = lora_linear_quant(m.v_proj, weight_quant, act_quant, quantize_bmm_input)
            m.o_proj = lora_linear_quant(m.o_proj, weight_quant, act_quant, quantize_bmm_input)
    return model

class Trainer:
    def __init__(self, model, data_module, tokenizer, device, batch_size=64, epochs=10):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.all_metrics = {}
        self.trainer = transformers.Seq2SeqTrainer(
                model=model,
                train_dataset = data_module["train_dataset"],
                eval_dataset = data_module["eval_dataset"],
                args=transformers.Seq2SeqTrainingArguments (
                    per_device_train_batch_size = batch_size,
                    gradient_accumulation_steps = 4,
                    warmup_steps=100,
                    # max_steps=200,
                    num_train_epochs = epochs,
                    learning_rate=1e-3,
                    # logging_steps=10,
                    output_dir='outputs'
                ),
            data_collator=data_module["data_collator"]
        )

    def train(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.all_metrics.update(metrics)
    def evaluate(self):
        metrics = self.trainer.evaluate(metric_key_prefix="eval")
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        self.all_metrics.update(metrics)
        return self.all_metrics



if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    epochs = args.epochs

    act_quant = "None" if args.act_quant == "None" else f"{args.act_quant}_absmax"
    weight_quant = "None" if args.weight_quant == "None" else f"{args.weight_quant}_absmax"
    model = build_lora_model(model_name)
    tokenizer = build_tokenizer(model_name, model)
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    if args.full:
        model = build_lora_model(model_name)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)

        if args.test:
            acc_fp32 = trainer.evaluate()
            print(f'Original model (fp32) before fine tuning accuracy: {acc_fp32}')
        if args.train:
            trainer.train()
            acc_fp32 = trainer.evaluate()
            print(f'Original model (fp32) after fine tuning accuracy: {acc_fp32}')

    if args.w8a8:
        model = build_lora_model(model_name)
        model = quantize_lora_model(model, act_quant=act_quant, weight_quant=weight_quant)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)

        if args.test:
            acc_w8a8 = trainer.evaluate()
            print(f'Naive W8A8 quantized model before fine tuning accuracy: {acc_w8a8}')

        if args.train:
            trainer.train()
            acc_w8a8 = trainer.evaluate()
            print(f'Naive W8A8 quantized model after fine tuning accuracy: {acc_w8a8}')

    # model = build_lora_model(model_name)
    # model = quantize_lora_model(model, act_quant="None", weight_quant=weight_quant)
    # trainer = Trainer(model, train_dataset, test_dataset, tokenizer, 'cuda', batch_size, epochs)
    #
    # acc_w8 = trainer.evaluate()
    # print(f'w8 model before fine tuning accuracy: {acc_w8}')
    #
    # trainer.train()
    #
    # acc_w8 = trainer.evaluate()
    # print(f'w8 model after fine tuning accuracy: {acc_w8}')

    if args.smooth:
        model = build_lora_model(model_name)
        act_scales = torch.load('act_scales/' + args.file_name)  # it is generated before test
        smooth_lm(model, act_scales, 0.5)
        model = quantize_lora_model(model, act_quant=act_quant, weight_quant=weight_quant)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_smooth = trainer.evaluate()
            print(f'SmoothQuant W8A8 quantized model before fine tuning accuracy: {acc_smooth}')

        if args.train:
            trainer.train()
            acc_smooth = trainer.evaluate()
            print(f'SmoothQuant W8A8 quantized model after fine tuning accuracy: {acc_smooth}')

    if args.simple_clip:
        act_quant = "None" if args.act_quant == "None" else f"{args.act_quant}_simple_clip"
        weight_quant = "None" if args.weight_quant == "None" else f"{args.weight_quant}_simple_clip"

        filename = f"{args.strategy}_{args.file_name}"
        model = build_lora_model(model_name)
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        model = lora_outlier_model_v2(model, act_quant=act_quant, weight_quant=weight_quant,outlier_dict=outlier_indices,)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')

    if args.clip:
        act_quant = "None" if args.act_quant == "None" else f"{args.act_quant}_clip"
        weight_quant = "None" if args.weight_quant == "None" else f"{args.weight_quant}_clip"

        filename = f"{args.strategy}_{args.file_name}"
        model = build_lora_model(model_name)
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        model = lora_outlier_model_v2(model, act_quant=act_quant, weight_quant=weight_quant,outlier_dict=outlier_indices,)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')

    if args.clip_v2:
        act_quant = "None" if args.act_quant == "None" else f"{args.act_quant}_clip_v2"
        weight_quant = "None" if args.weight_quant == "None" else f"{args.weight_quant}_clip_v2"

        filename = f"{args.strategy}_{args.file_name}"
        model = build_lora_model(model_name)
        outlier_indices = torch.load('outlier_idx/'+filename)  # it is generated before test
        model = lora_outlier_model_v2(model, act_quant=act_quant, weight_quant=weight_quant,outlier_dict=outlier_indices,)
        trainer = Trainer(model, data_module, tokenizer, 'cuda', batch_size, epochs)
        if args.test:
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model before fine tuning accuracy: {acc_lora}')

        if args.train:
            trainer.train()
            acc_lora = trainer.evaluate()
            print(f'LoRA {args.strategy} W8A8 quantized model after fine tuning accuracy: {acc_lora}')