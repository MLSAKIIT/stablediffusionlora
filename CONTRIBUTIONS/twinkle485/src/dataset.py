import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, augment=False):
        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.captions_dir = os.path.join(dataset_dir, 'ImageCaptions')
        self.augment = augment

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

        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        caption_path = os.path.join(self.captions_dir, os.path.splitext(img_name)[0] + '.txt')

        try:
            image = Image.open(img_path).convert('RGB')
            if self.augment:
                image = self.augmentation_transform(image)
            elif self.transform:
                image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        return image, caption

dataset_dir = '/content/Dataset'
dataset = CustomDataset(dataset_dir, augment=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for images, captions in dataloader:
    print(f"Image batch shape: {images.shape}")
    print(captions)