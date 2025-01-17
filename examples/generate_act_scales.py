import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales
from smoothquant.model import build_lora_model, build_tokenizer

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name facebook/opt-1.3b, meta-llama/Llama-2-7b-hf')
    parser.add_argument('--file-name', type=str,
                        default='opt-1.3b.pt', help='model name: opt-1.3b.pt, Llama-2-7b-hf.pt')
    # parser.add_argument('--output-path', type=str, default='act_scales/opt-1.3b.pt',
    #                     help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='OpenAssistant/oasst1',
                        help='location of the calibration dataset, we use the validation set of the oasst1 validation dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if "llama" in args.model_name:
        model_name = args.model_name
        model = build_lora_model(model_name)
        tokenizer = build_tokenizer(model_name, model)
    else:
        model, tokenizer = build_model_and_tokenizer(args.model_name)

    # model = build_lora_model(args.model_name) # comment it when use PTQ

    # if not os.path.exists(args.dataset_path):
    #     print(f'Cannot find the dataset at {args.dataset_path}')
    #     print('Please download the Pile dataset and put the validation set at the path')
    #     print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
    #     raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname("act_scales/"), exist_ok=True)
    torch.save(act_scales, "act_scales/" + args.file_name)


if __name__ == '__main__':
    main()
