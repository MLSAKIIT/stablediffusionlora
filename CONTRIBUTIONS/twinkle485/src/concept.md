# Data Augmentation Techniques and Enhancements to Training Loop Efficiency

## Purpose:

This update focuses on improving the training loop by integrating a more advanced learning rate scheduling mechanism for optimized training efficiency, and gradient clipping to stabilize training. These modifications ensure that the model can handle various training challenges more effectively.

## Changes made:

# 1. Training Loop Enhancements (`train.py`)

The training loop has been significantly improved to increase **stability**, **efficiency**, and **scalability**. Below are the key changes made to enhance the training process:

## Key Features:

1. **Learning Rate Scheduler (CosineAnnealingLR)**

   - A **Cosine Annealing Learning Rate Scheduler** was added to dynamically adjust the learning rate during training. 
   - This scheduler helps avoid overshooting by gradually reducing the learning rate as the training progresses, ensuring a more refined convergence.
   - As training continues, the learning rate smoothly decreases to a small value (`eta_min`), allowing the model to fine-tune during the final stages of training.

   **Code Implementation:**
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR
   
   scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

2. **Gradient Clipping**

    - **Gradient Clipping** was implemented to cap gradients at a maximum norm of 1.0. 
    - This prevents the gradients from growing too large, a phenomenon known as "exploding gradients," which can cause instability during training.

    **Code Implementation:**
    ```python
    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)


# 2. Data Augmentation

To improve the model's generalization and make it more robust to variations, several **data augmentation techniques** have been added to the dataset preprocessing pipeline.

These augmentations are applied to the images during training to introduce random transformations, helping the model learn from a more diverse set of data without needing additional images.

### Key Augmentation Techniques:
- **RandomResizedCrop**: Randomly crops the image to different sizes and aspect ratios and then resizes it to a fixed size.
- **RandomHorizontalFlip**: Horizontally flips the image with a probability of 0.5.
- **RandomRotation**: Rotates the image randomly within a specified degree range.
- **ColorJitter**: Randomly changes the brightness, contrast, saturation, and hue of the image.
- **GaussianBlur**: Applies a Gaussian blur with a random sigma value to the image.
- **RandomAdjustSharpness**: Adjusts the sharpness of the image with a given probability.
- **RandomAffine**: Applies random affine transformations, including translation, scaling, and shearing.
- **RandomErasing**: Randomly erases a portion of the image, simulating occlusions and missing data.

### Code Implementation:
```python
self.augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.05, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])
