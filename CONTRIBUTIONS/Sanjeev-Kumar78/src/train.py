import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def train_loop(dataloader, val_loader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs, patience=5):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        unet.train()
        
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
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)
        progress_bar.close()
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Validate the model
        val_loss = validate(val_loader, unet, text_encoder, vae, noise_scheduler, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete!")

def validate(val_loader, unet, text_encoder, vae, noise_scheduler, device):
    unet.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
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
            val_loss += loss.item()
    
    return val_loss / len(val_loader)