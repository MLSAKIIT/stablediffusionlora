import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
def train_loop(dataloader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, captions = batch
            images = images.to(device)
            
            # Encode text
            text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            
            # Prepare latents
            latents = vae.encode(images).latent_dist.sample().detach()
            latents = latents * 0.18215
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Compute the loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training complete!")