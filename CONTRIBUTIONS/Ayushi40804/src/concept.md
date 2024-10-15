# Applying LoRA to Convolutional Layers

## Overview

Enhance Pokémon image generation by incorporating convolutional layers and improving parameterization for better spatial hierarchies and patterns.

## Change in Architecture : Implementation of LoRALayerConv2d

The `LoRALayerConv2d` class in `lora.py` is designed to introduce low-rank adaptations to convolutional layers, enhancing their ability to capture complex patterns in the data.

```python
class LoRALayerConv2d(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, rank=4, alpha=1, stride=1, padding=0):
      super().__init__()
      self.rank = rank
      self.alpha = alpha
      self.conv1 = nn.Conv2d(in_channels, rank, kernel_size, stride, padding)
      self.conv2 = nn.Conv2d(rank, out_channels, 1)

   def forward(self, x):
      return self.alpha * self.conv2(self.conv1(x))
```

## How LoRA is Applied

 ### Initialization of LoRA-enhanced Convolutional Layer:

 - The `LoRALayerConv2d` class is initialized with the parameters of the original convolutional layer `conv`, including `in_channels`, `out_channels`, `kernel_size`, `stride`, and `padding`.
 - Two low-rank matrices, `lora_A` and `lora_B`, are created as parameters. These matrices are initialized with zeros and then adjusted using specific initialization methods (`kaiming_uniform_` for `lora_A` and `zeros_` for `lora_B`).
 ### Forward Pass:
 - In the `forward` method of the `LoRACONV2d` class, the original convolutional layer (`self.conv`) is applied to the input `x`.
 - The LoRA-enhanced convolutional layer (`self.lora`) is also applied to the input `x`.
 - The outputs of the original convolutional layer and the LoRA-enhanced layer are summed together to produce the final output.

## Which Layers are Involved
### Original Convolutional Layer (`self.conv`):

 - This is the standard convolutional layer that was originally part of the model. It performs the usual convolution operation on the input data.
### LoRA-enhanced Convolutional Layer (`self.lora`):

 - This layer is an instance of the LoRALayerConv2d class. It performs a low-rank adaptation of the convolution operation using the matrices lora_A and lora_B.

## Integration with Existing Architecture

The `LoRACONV2d` class is introduced to combine the original convolutional layer with the low-rank adaptation layer.
```bash
+----------------------------+
|       Input Image          |
+----------------------------+
              |
              v
+----------------------------+
|      Original Conv Layer   |
|      (Convolutional Layer) |
+----------------------------+
              |
              |---------------------+
              |                     |
              v                     v
+----------------------------+   +----------------------------+
|      LoRALayer (lora_A)   |   |   Original Conv Output     |
| (Low-Rank Adaptation Layer)|   |   (Without LoRA)          |
+----------------------------+   +----------------------------+
              |                     |
              |                     |
              v                     v
+----------------------------+   +----------------------------+
|    LoRA Output (lora_A)   |   |  Combined Conv Output      |
|  (Low-Rank Adaptation)     |   | (Original + LoRA Output)  |
+----------------------------+   +----------------------------+
              |                     |
              |                     |
              +---------------------+
              |
              v
+----------------------------+
|      Final Output          |
|   (Feature Map for Next    |
|      Network Layer)        |
+----------------------------+
```