import torch
import os
from utils import load_lora_weights
from diffusers import StableDiffusionPipeline

def load_trained_model(model_path, lora_weights_path, device):
    # Load the pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to(device)
    # Load the LoRA weights
    load_lora_weights(pipe.unet, lora_weights_path)
    return pipe

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "CompVis/stable-diffusion-v1-4"  # Base model path

    lora_weights_path = "CONTRIBUTIONS\Sanjeev-Kumar78\src\Models\lora_weights.pt"  # LoRA weights path
    pipe = load_trained_model(model_path, lora_weights_path, device)

    prompts = [
        "A Bulbasaur", "Pokemon Bulbasaur", "Green Bulbasaur in HD", "Bulbasaur", "Bulbasaur"
    ]
    Images = []
    for prompt in prompts:
        image = generate_image(prompt, pipe)
        Images.append(image)
    save_images(Images, "CONTRIBUTIONS\Sanjeev-Kumar78\src\output_images")
