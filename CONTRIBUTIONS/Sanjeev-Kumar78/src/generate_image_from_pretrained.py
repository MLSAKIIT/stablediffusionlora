import torch
import os
from lora import LoRALayer
from diffusers import StableDiffusionPipeline

def merge_lora_weights(model, paths, device):
    merged_state_dict = {}
    for path in paths:
        lora_state_dict = torch.load(path, weights_only=False, map_location=device)
        for key, value in lora_state_dict.items():
            if key in merged_state_dict:
                with torch.no_grad():
                    merged_state_dict[key] += value
            else:
                merged_state_dict[key] = value

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.lora_A.data = merged_state_dict[f"{name}.lora_A"]
            module.lora_B.data = merged_state_dict[f"{name}.lora_B"]
    print(f"LoRA weights merged from {paths}")

def load_trained_model(model_path, lora_weights_path, device):
    # Load the pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe = pipe.to(device)
    # Load the LoRA weights
    merge_lora_weights(pipe.unet, lora_weights_path, device)
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

    lora_weights_path = ["D:\CODE\Hacktoberfest'24\MLSA_KIIT\stablediffusionlora\CONTRIBUTIONS\Sanjeev-Kumar78\src\Models\lora_weights_fold_0.pt","D:\CODE\Hacktoberfest'24\MLSA_KIIT\stablediffusionlora\CONTRIBUTIONS\Sanjeev-Kumar78\src\Models\lora_weights_fold_1.pt"]  # LoRA weights path
    pipe = load_trained_model(model_path, lora_weights_path, device)

    prompts = [
        "A Bulbasaur", "Pokemon Bulbasaur", "Green Bulbasaur in HD", "Bulbasaur", "Bulbasaur"
    ]
    Images = []
    for prompt in prompts:
        image = generate_image(prompt, pipe)
        Images.append(image)
    save_images(Images, "CONTRIBUTIONS\Sanjeev-Kumar78\src\output_images")
