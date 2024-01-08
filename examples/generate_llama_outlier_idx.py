import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.model import build_lora_model, build_tokenizer
from smoothquant.calibration import get_act_outlier_idx

def build_model_and_tokenizer(model_name):
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    end = time.time() - start
    print(f"Loading tokenizer Time : {round(end, 2)}s")

    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    end = time.time() - start
    print(f"Loading model Time : {round(end, 2)}s")
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='meta-llama/Llama-2-7b-hf', help='model name')
    parser.add_argument('--file-name', type=str,
                        default='Llama-2-7b-hf.pt', help='model name')
    # parser.add_argument('--output-path', type=str, default='outlier_idx/opt-1.3b.pt',
    #                     help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='OpenAssistant/oasst1',
                        help='location of the calibration dataset, we use the validation set of the oasst1 validation dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--strategy', type=str,
                        default='IQR', help='outlier detection strategy')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model_name = args.model_name
    model = build_lora_model(model_name)
    tokenizer = build_tokenizer(model_name, model)

    # if not os.path.exists(args.dataset_path):
    #     print(f'Cannot find the dataset at {args.dataset_path}')
    #     print('Please download the Pile dataset and put the validation set at the path')
    #     print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
    #     raise FileNotFoundError
    outlier_idx, mean_upper_bound = get_act_outlier_idx(model, tokenizer, args.dataset_path, args.num_samples, args.seq_len, strategy=args.strategy)

    os.makedirs(os.path.dirname("outlier_idx/"), exist_ok=True)
    os.makedirs(os.path.dirname("upper_bound/"), exist_ok=True)
    torch.save(outlier_idx, f"outlier_idx/{args.strategy}_"+ args.file_name)
    torch.save(mean_upper_bound, f"upper_bound/{args.strategy}_"+ args.file_name)

if __name__ == '__main__':
    main()
