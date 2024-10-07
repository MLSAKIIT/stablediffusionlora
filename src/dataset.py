import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.captions_dir = os.path.join(dataset_dir, 'ImageCaptions')
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: x.to(torch.float32))
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
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
    
        # Check if the caption file exists
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")
    
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
    
        return image, caption
