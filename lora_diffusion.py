import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank
        self.rank = rank

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scale

class LoRANetwork(nn.Module):
    def __init__(self, base_model, rank=4, alpha=1):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleDict()
        
        # TODO: Implement LoRA layer injection into the base model

    def forward(self, x):
        # TODO: Implement forward pass with LoRA layers
        pass

def inject_trainable_lora(model, rank=4, alpha=1):
    # TODO: Implement LorA injction into the model
    pass
