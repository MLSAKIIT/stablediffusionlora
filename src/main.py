from dataset import CustomDataset
from lora import apply_lora_to_model
from train import train_loop
from utils import save_lora_weights
from generate import generate_image, save_images
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Load model components
    
    dtype = torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    
    vae = pipe.vae.to(device, dtype)
    text_encoder = pipe.text_encoder.to(device, dtype)
    unet = apply_lora_to_model(pipe.unet).to(device, dtype)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    dataset = CustomDataset("Dataset")  # Update this path as needed
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


 
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch}")  
        if i >= 5:  
            break

    
    # Training
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5) #LR should be 1e-5 for small datasets of 10-20 images. using a higher LR will cause serious overfitting. 1e-6 may be used for larger datasets.
    num_epochs = 20
    train_loop(dataloader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs)
    
    # Save LorA weights
    save_lora_weights(unet, "lora_weights.pt")
    
    # Generate samples(update prompts as per your finetuning concept)
    prompts = [
        "A Bulbasaur","Pokemon Bulbasaur", "Green Bulbasaur in HD", "Bulbasaur", "Bulbasaur"
    ]
    pipe.unet = unet
    images = [generate_image(prompt, pipe) for prompt in prompts]
    save_images(images, "output_images")

if __name__ == "__main__":
    main()
