import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scale

def add_lora_to_linear(linear, rank=4, alpha=1):
    lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    return lambda x: linear(x) + lora(x)

def apply_lora_to_model(model, rank=4, alpha=1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, add_lora_to_linear(module, rank, alpha))
    return model