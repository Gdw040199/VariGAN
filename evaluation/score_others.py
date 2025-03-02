import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim


# PSNR calculation function
def calculate_psnr(img1, img2, max_pixel=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# SSIM calculation function
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)


# FIR calculation class
class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, aux_logits=True).to(device)
        self.model.eval()

    def get_activations(self, image_batch):
        with torch.no_grad():
            activations = self.model(image_batch)
        return activations.cpu().numpy()

    def calculate_fid(self, real_activations, fake_activations):
        mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)

        ss = np.sum((mu1 - mu2) ** 2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ss + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid


# load image function and batch processing
def load_images(image_folder, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
    batches = []

    for i in range(0, len(image_files), batch_size):
        batch = []
        for f in image_files[i:i + batch_size]:
            img = Image.open(os.path.join(image_folder, f)).convert('RGB')
            batch.append(transform(img))
        batches.append(torch.stack(batch))

    return batches


def main():
    dir = 'your_image_folder'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fid_calculator = FIDCalculator(device)

    batches = load_images(dir)

    psnr_values = []
    ssim_values = []

    for batch in batches:
        for img in batch:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            psnr_values.append(calculate_psnr(img_np, img_np))
            ssim_values.append(calculate_ssim(img_np, img_np))

    features = []

    for batch in batches:
        features.append(fid_calculator.get_activations(batch.to(device)))

    fid_score = fid_calculator.calculate_fid(features[0], features[0])

    print(f"PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"SSIM: {np.mean(ssim_values):.4f}")
    print(f"FID: {fid_score:.2f}")


if __name__ == "__main__":
    main()
