# ============================================================
# STEP 1: Project Setup
# Commit: "Initial commit - project setup"
# ============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create output directory
os.makedirs("./outputs", exist_ok=True)
print("Project setup complete.")


# ============================================================
# STEP 2: Install Required Libraries (done via pip)
# Commit: "Installed required libraries"
# Run: pip install torch torchvision numpy matplotlib pillow opencv-python
# ============================================================


# ============================================================
# STEP 3: Load and Preprocess the Satellite-to-Map Dataset
# Commit: "Loaded and preprocessed image datasets for Pix2Pix GAN"
# ============================================================

# Custom dataset that loads paired satellite/map images from a single folder
# Each image is a side-by-side pair (satellite | map), as used in pix2pix paper
class Pix2PixDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = [os.path.join(root, f) for f in os.listdir(root)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        # Left half = satellite, right half = map
        satellite = img.crop((0, 0, w // 2, h))
        map_img = img.crop((w // 2, 0, w, h))
        if self.transform:
            satellite = self.transform(satellite)
            map_img = self.transform(map_img)
        return satellite, map_img

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# NOTE: Download dataset from https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
# Place images in ./data/maps/train/
DATA_DIR = "./data/maps/train"
if not os.path.exists(DATA_DIR):
    print(f"[!] Dataset not found at {DATA_DIR}. Creating dummy data for demo.")
    os.makedirs(DATA_DIR, exist_ok=True)
    # Create a dummy paired image for demo
    dummy = Image.fromarray(np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8))
    for i in range(20):
        dummy.save(f"{DATA_DIR}/dummy_{i}.jpg")

dataset = Pix2PixDataset(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Display a sample
sat_sample, map_sample = dataset[0]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sat_sample.permute(1, 2, 0) * 0.5 + 0.5)
axes[0].set_title("Satellite")
axes[0].axis("off")
axes[1].imshow(map_sample.permute(1, 2, 0) * 0.5 + 0.5)
axes[1].set_title("Map")
axes[1].axis("off")
plt.suptitle("Sample Dataset Pair")
plt.tight_layout()
plt.savefig("./outputs/sample.png")
plt.close()
print("Dataset loaded. Sample saved to ./outputs/sample.png")




# ============================================================
# STEP 4: Implement Pix2Pix Generator (U-Net) and Discriminator (PatchGAN)
# Commit: "Implemented Pix2Pix Generator and Discriminator models"
# ============================================================

# Generator: Simple U-Net style encoder-decoder
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)                      # 256->128
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128))  # 128->64
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256)) # 64->32
        # Decoder with skip connections (U-Net style)
        self.dec3 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128)) # 32->64
        self.dec2 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64))  # 64->128 (256=128+128 skip)
        self.dec1 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())             # 128->256

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # Skip connection
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # Skip connection
        return d1

# Discriminator: PatchGAN (classifies patches as real/fake)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),                                        # Input: sat+map concatenated
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()                          # Patch output
        )

    def forward(self, sat, map_img):
        x = torch.cat([sat, map_img], dim=1)  # Concatenate input pair
        return self.model(x)

# Initialize models
G = Generator()
D = Discriminator()
print("Models initialized.")
print(f"  Generator params: {sum(p.numel() for p in G.parameters()):,}")
print(f"  Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
