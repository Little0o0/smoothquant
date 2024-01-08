from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from typing import Dict


DEFAULT_PAD_TOKEN = "[PAD]"

def build_lora_model(model_name):
    if "opt" in model_name.lower():
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='auto')

        peft_config = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif "llama" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float32, device_map='auto')

        setattr(model, 'model_parallel', True)
        setattr(model, 'is_parallelizable', True)

        peft_config = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        raise Exception(f"Model {model_name} is not supported yet !! ")
    return model


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def build_tokenizer(model_name, model):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in model_name.lower() else None,  # Needed for HF name change
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if "llama" in model_name.lower():
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    return tokenizer