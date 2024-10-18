import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn as n
import math
from transformers import CLIPTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR 

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def train_loop(dataloader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs):

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in dataloader:
            images, captions = batch
            images = images.to(device)

            # Encode text
            text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

            # Prepare latents
            latents = vae.encode(images).latent_dist.sample().detach()
            latents = latents * 0.18215

            # Add noise to latent
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)
        progress_bar.close()
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
    print("Training complete!")