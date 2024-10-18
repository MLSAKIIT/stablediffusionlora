import torch
from lora import LoRALayer


def save_lora_weights(model, path):
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path}")

def load_lora_weights(model, path):
    lora_state_dict = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    print(f"LoRA weights loaded from {path}")