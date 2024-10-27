import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import lpips

def load_image(image_path):
    """Loads and preprocesses an image for evaluation."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def calculate_fid(real_images_path: str, generated_images_path: str):
    """
    Calculates the Frechet Inception Distance (FID) between real and generated images.
    
    Parameters:
    -----------
    real_images_path : str
        Path to the directory containing real images.
    generated_images_path : str
        Path to the directory containing generated images.
        
    Returns:
    --------
    float
        FID score between the two sets of images.
    """
    fid = FrechetInceptionDistance(feature=64)

    # Load real and generated images
    real_image = load_image(real_images_path)
    generated_image = load_image(generated_images_path)

    # Update FID with the images
    fid.update(real_image, real=True)
    fid.update(generated_image, real=False)

    fid_score = fid.compute()
    print(f"FID Score: {fid_score}")
    return fid_score

def calculate_lpips(image_1_path: str, image_2_path: str):
    """
    Calculates the LPIPS (Learned Perceptual Image Patch Similarity) between two images.
    
    Parameters:
    -----------
    image_1_path : str
        Path to the first image.
    image_2_path : str
        Path to the second image.
        
    Returns:
    --------
    float
        LPIPS score between the two images.
    """
    image1 = load_image(image_1_path)
    image2 = load_image(image_2_path)

    lpips_score = lpips(image1, image2)
    print(f"LPIPS Score: {lpips_score}")
    return lpips_score

if __name__ == "__main__":
    # Example usage
    real_image_path = "./real_image.jpg"
    generated_image_path = "./generated_image.jpg"

    # Calculate FID
    fid_score = calculate_fid(real_image_path, generated_image_path)

    # Calculate LPIPS
    lpips_score = calculate_lpips(real_image_path, generated_image_path)
