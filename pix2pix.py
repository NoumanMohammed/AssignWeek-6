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


# ============================================================
# STEP 5: Train Pix2Pix GAN
# Commit: "Trained Pix2Pix GAN for satellite-to-map translation"
# ============================================================

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Track losses for plotting
d_losses, g_losses, l1_losses = [], [], []

epochs = 10
print(f"\nStarting training for {epochs} epochs...")

for epoch in range(epochs):
    epoch_d, epoch_g, epoch_l1 = 0, 0, 0
    for sat_imgs, map_imgs in dataloader:
        batch = sat_imgs.size(0)
        patch = D(sat_imgs, map_imgs).shape[-1]       # Dynamically get patch size
        real_label = torch.ones(batch, 1, patch, patch)   # PatchGAN real target
        fake_label = torch.zeros(batch, 1, patch, patch)  # PatchGAN fake target

        # --- Train Discriminator ---
        fake_map = G(sat_imgs)
        d_real = adversarial_loss(D(sat_imgs, map_imgs), real_label)
        d_fake = adversarial_loss(D(sat_imgs, fake_map.detach()), fake_label)
        d_loss = (d_real + d_fake) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        g_adv = adversarial_loss(D(sat_imgs, fake_map), real_label)  # Fool discriminator
        g_l1 = l1_loss(fake_map, map_imgs) * 10                       # Pixel-level similarity (lambda=10)
        g_loss = g_adv + g_l1

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        epoch_d += d_loss.item()
        epoch_g += g_adv.item()
        epoch_l1 += g_l1.item()

    n = len(dataloader)
    d_losses.append(epoch_d / n)
    g_losses.append(epoch_g / n)
    l1_losses.append(epoch_l1 / n)
    print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_losses[-1]:.4f} | G_loss: {g_losses[-1]:.4f} | L1: {l1_losses[-1]:.4f}")

print("Training complete!")


# ============================================================
# STEP 6: Evaluate and Visualize Translated Images
# Commit: "Evaluated Pix2Pix and visualized image translation"
# ============================================================

G.eval()
with torch.no_grad():
    test_sat, test_map = dataset[0]
    test_sat = test_sat.unsqueeze(0)
    translated = G(test_sat).squeeze().permute(1, 2, 0).numpy()
    translated = np.clip(translated * 0.5 + 0.5, 0, 1)  # Denormalize

# --- Output 1: Image Translation Result (sample.jpg) ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(test_sat.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[0].set_title("Input (Satellite)")
axes[0].axis("off")
axes[1].imshow(translated)
axes[1].set_title("Generated (Map)")
axes[1].axis("off")
axes[2].imshow(test_map.permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[2].set_title("Ground Truth (Map)")
axes[2].axis("off")
plt.suptitle("Pix2Pix: Satellite-to-Map Translation")
plt.tight_layout()
plt.savefig("./outputs/sample.jpg")
plt.close()
print("Saved: ./outputs/sample.jpg")

# --- Output 2: Training Loss Curves ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), d_losses, label="Discriminator Loss", marker='o')
plt.plot(range(1, epochs+1), g_losses, label="Generator Adversarial Loss", marker='s')
plt.plot(range(1, epochs+1), l1_losses, label="Generator L1 Loss", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pix2Pix GAN Training Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./outputs/loss_curves.png")
plt.close()
print("Saved: ./outputs/loss_curves.png")

# --- Output 3: Pixel-Level Error Heatmap (L1 per pixel) ---
gt_map = test_map.permute(1, 2, 0).numpy() * 0.5 + 0.5
l1_error = np.abs(translated - gt_map).mean(axis=2)  # Mean across channels

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(translated)
axes[0].set_title("Generated Map")
axes[0].axis("off")
axes[1].imshow(gt_map)
axes[1].set_title("Ground Truth Map")
axes[1].axis("off")
im = axes[2].imshow(l1_error, cmap='hot')
axes[2].set_title("Pixel Error Heatmap")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.suptitle("Pixel-Level L1 Error: Generated vs Ground Truth")
plt.tight_layout()
plt.savefig("./outputs/error_heatmap.png")
plt.close()
print("Saved: ./outputs/error_heatmap.png")

print("\nAll outputs saved to ./outputs/")
print("Done! Push to GitHub with: 'Evaluated Pix2Pix and visualized image translation'")