
# Stable Diffusion 1.4 Fine-tuning with LoRA: Technical Implementation

[![Hacktoberfest 2024](https://img.shields.io/badge/Hacktoberfest-2024-orange.svg)](https://hacktoberfest.com/)

This document outlines the technical implementation of fine-tuning Stable Diffusion 1.4 using Low-Rank Adaptation (LoRA). It provides a detailed guide for beginners to understand and contribute to the project.

## Project Structure

```
sd-lora-finetuning/
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
├── requirements.txt
├── src/  (Example implementation)
│   ├── Dataset/
│   │   ├── ImageCaptions/
│   │   │   └── example1.txt
│   │   └── Images/
│   │       └── example1.png
│   ├── dataset.py
│   ├── generate.py
│   ├── lora.py
│   ├── main.py
|   |── scraping.py
│   ├── train.py
│   └── utils.py
└── CONTRIBUTIONS/
    └── Example1/
        ├── Dataset/
        │   ├── ImageCaptions/
        │   │   └── example1.txt
        │   └── Images/
        │       └── example1.png
        └── src/
            ├── dataset.py
            ├── generate.py
            ├── lora.py
            ├── main.py
            |── scraping.py
            ├── train.py
            └── utils.py

```

- `src/`: Contains the example implementation (refer to this for your contribution)
- `CONTRIBUTIONS/`: Directory where participants should add their implementations
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`: Guidelines and help regarding contributing(MUST READ!)
- Other files in the root directory are for project documentation and setup

## Technical Overview

### 1. LoRA Implementation (`lora.py`)

LoRA is implemented as follows:

a) `LoRALayer` class:
   ```python
   class LoRALayer(nn.Module):
       def __init__(self, in_features, out_features, rank=4, alpha=1):
           super().__init__()
           self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
           self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
           self.scale = alpha / rank
           nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
           nn.init.zeros_(self.lora_B)

       def forward(self, x):
           return (x @ self.lora_A.T @ self.lora_B.T) * self.scale
   ```

b) `apply_lora_to_model` function:
   ```python
   def apply_lora_to_model(model, rank=4, alpha=1):
       for name, module in model.named_modules():
           if isinstance(module, nn.Linear):
               lora_layer = LoRALayer(module.in_features, module.out_features, rank, alpha)
               setattr(module, 'lora', lora_layer)
       return model
   ```

Key concept: LoRA adds trainable low-rank matrices to existing layers, allowing for efficient fine-tuning.

### 2. Dataset Handling (`dataset.py`)

The `CustomDataset` class:
```python
class CustomDataset(Dataset):
    def __init__(self, img_dir, caption_dir=None, transform=None):
        self.img_dir = img_dir
        self.caption_dir = caption_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.caption_dir:
            caption_path = os.path.join(self.caption_dir, self.images[idx].rsplit('.', 1)[0] + '.txt')
            with open(caption_path, 'r') as f:
                caption = f.read().strip()
        else:
            caption = ""

        return image, caption
```

### 3. Training Process (`train.py`)

The `train_loop` function implements the core training logic:

```python
def train_loop(dataloader, unet, text_encoder, vae, noise_scheduler, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            images, captions = batch
            latents = vae.encode(images.to(device)).latent_dist.sample().detach()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            text_embeddings = text_encoder(captions)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

Key concept: We're training the model to denoise latent representations, conditioned on text embeddings.

### 4. Image Generation (`generate.py`)

```python
def generate_image(prompt, pipeline, num_inference_steps=50):
    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
    return image
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sd-lora-finetuning.git
   cd sd-lora-finetuning
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset Generation
If you want to scrape images and captions from the web, you can use the `scraping.py` script, which uses BING API to download images and captions which can be used for training. `Dataset` folder should be created in the `src` directory to store the images and captions.
 - How To Use:
   - Create a `.env` file in the root directory and add the following:
     ```
     API_KEY="YOUR_BING_API_KEY"
     ```
    - You can get the API key from the Azure portal [Create](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api).
    - Replace this "YOUR_BING_API_KEY" with your API key.
    - Change the `query` in the [`scraping.py`](src\scraping.py) file to the desired search query.
    - Change the path in the [`scraping.py`](src\scraping.py) file to the desired path where you want to store the images and captions.
    - Run the `scraping.py` file to download the images and captions.
  
  **Note:** The Image Caption will be the same as the Search Query. So, make sure to use the search query that you want to use as the caption of the image while training.

## Contributing

1. Fork the repository and clone your fork.
2. Create a new folder in the `CONTRIBUTIONS` directory with your username or project name.
3. Implement your version of the LoRA fine-tuning following the structure in the `src` directory.
4. Ensure you include a `Dataset` folder with example images and captions.
5. Create a pull request with your contribution.

Refer to the `src` directory for an example of how to structure your contribution.

Refer to `CONTRIBUTING.md` for a detailed overview, if you're a beginner!

## Technical Deep Dive

### LoRA Mechanism

LoRA adapts the model by injecting trainable rank decomposition matrices into existing layers:

1. For a layer with weight W, LoRA adds BA where B ∈ R^(d×r) and A ∈ R^(r×k)
2. The output is computed as: h = W*x + BA*x
3. Only A and B are trained, keeping the original weights W frozen

This is implemented in the `LoRALayer` class:

```python
def forward(self, x):
    return (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

### Training Objective

The model is trained to predict the noise added to the latent representation:

1. Images are encoded to latent space: z = Encode(x)
2. Noise is added: z_noisy = z + ε
3. The model predicts the noise: ε_pred = Model(z_noisy, t, text_embedding)
4. Loss is computed: L = MSE(ε_pred, ε)

This is implemented in the training loop:

```python
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
loss = F.mse_loss(noise_pred, noise)
```

By learning to denoise, the model implicitly learns to generate images conditioned on text.

## Customization and Extension

Customimzation and uniqueness is expected from each contributor.

- Feel free to modify `LoRALayer` in `lora.py` to experiment with different LoRA architectures
- Adjust the U-Net architecture in `main.py` by modifying which layers receive LoRA
- Implement additional training techniques in `train.py` (e.g., gradient clipping, learning rate scheduling)

## Resources

- LoRA paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Stable Diffusion: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Hugging Face Diffusers: [Documentation](https://huggingface.co/docs/diffusers/index)


