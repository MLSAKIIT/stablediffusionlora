import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import CustomDataset
from lora import apply_lora_to_model
from train import train_loop
from utils import save_lora_weights
from generate import generate_image, save_images
from diffusers import StableDiffusionPipeline, DDPMScheduler
from concurrent.futures import ThreadPoolExecutor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    vae = pipe.vae.to(device, dtype)
    text_encoder = pipe.text_encoder.to(device, dtype)
    pipe.unet = pipe.unet.to(dtype=torch.float32)
    unet = apply_lora_to_model(pipe.unet).to(device, dtype)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    dataset = CustomDataset("CONTRIBUTIONS\Sanjeev-Kumar78\src\Dataset")  # Update this path as needed
    k_folds = 2
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=1, shuffle=False)

        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=1e-5)
        num_epochs = 4

        train_loop(train_loader, val_loader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs, pipe=pipe)

        # Save LorA weights for each fold
        save_lora_weights(unet, f"CONTRIBUTIONS\Sanjeev-Kumar78\src\Models\lora_weights_fold_{fold}.pt")

    # Generate samples (update prompts as per your fine-tuning concept)
    prompts = [
        "A Bulbasaur", "Pokemon Bulbasaur", "Green Bulbasaur in HD", "Bulbasaur", "Bulbasaur"
    ]
    pipe.unet = unet

    # Function to generate and save images
    def generate_and_save_image(prompt):
        image = generate_image(prompt, pipe)
        return image

    # Use ThreadPoolExecutor to parallelize image generation
    with ThreadPoolExecutor(max_workers=2) as executor:
        images1 = list(executor.map(generate_and_save_image, prompts[:2]))
        images2 = list(executor.map(generate_and_save_image, prompts[2:]))
        images = images1 + images2

    save_images(images, "CONTRIBUTIONS\Sanjeev-Kumar78\src\output_images")

if __name__ == "__main__":
    main()