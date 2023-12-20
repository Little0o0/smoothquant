from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
import torch

def build_lora_model(model_name):
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='auto')

    peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model