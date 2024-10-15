import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=1):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1, dtype=torch.float32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), dtype=dtype))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scale


class LoRACONV2d(nn.Module):
    def __init__(self, conv, rank=4, alpha=1):
        super().__init__()
        self.conv = conv  # Store the original convolution layer
        self.lora = LoRALayerConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, rank, alpha,
                                     conv.stride, conv.padding)

    def forward(self, x):
        return self.conv(x) + self.lora(x)


class LoRALayerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, alpha=1, stride=1, padding=0, dtype=torch.float32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, in_channels, *kernel_size), dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_channels, rank, 1, 1), dtype=dtype))  # Keep this 1x1 for matching dimensions
        self.scale = alpha / rank
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Convolution with lora_A
        lora_output = nn.functional.conv2d(x, self.lora_A.view(-1, *self.lora_A.shape[1:]), stride=self.stride, padding=self.padding)

        # Using a pointwise convolution to combine with lora_B
        lora_output = lora_output.view(lora_output.shape[0], -1, lora_output.shape[2], lora_output.shape[3])  # Reshape for multiplication

        # Multiply with lora_B
        lora_output = nn.functional.conv2d(lora_output, self.lora_B, stride=1, padding=0)  # Pointwise convolution to reduce dimensions

        return lora_output * self.scale


def add_lora_to_linear(linear, rank=4, alpha=1):
    lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    return lambda x: linear(x) + lora(x)

def add_lora_to_conv(conv, rank=4, alpha=1):
    lora = LoRALayerConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, rank, alpha,
                            conv.stride, conv.padding)  # Added stride and padding
    return lambda x: conv(x) + lora(x)

def apply_lora_to_model(model, rank=4, alpha=1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            lora_layer = LoRALinear(module, rank, alpha)
            lora_layer = lora_layer.to(module.weight.device, module.weight.dtype)
            setattr(parent, child_name, lora_layer)
        elif isinstance(module, nn.Conv2d):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            lora_layer = LoRACONV2d(module, rank, alpha)
            lora_layer = lora_layer.to(module.weight.device, module.weight.dtype)
            setattr(parent, child_name, lora_layer)
    return model
