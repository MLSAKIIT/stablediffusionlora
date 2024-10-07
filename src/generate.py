import torch
from PIL import Image
import os

def generate_image(prompt, pipeline, num_inference_steps=50):
    try:
        with torch.no_grad():
            image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{output_dir}/generated_image_{i+1}.png")
    print(f"Images saved to {output_dir}")