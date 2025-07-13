"""
🧩 Customer Segmentation using ClusterGAN

This script performs unsupervised customer segmentation using a GAN-based approach.
Steps:
1. Loads RFM-scaled customer data
2. Trains a lightweight Generator–Discriminator GAN
3. Uses the Generator to create latent embeddings
4. Applies KMeans on the generated embeddings
5. Saves the trained KMeans model for deployment

Output: models/segmentation/clustergan_model.pkl
"""
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from preprocess import get_preprocessed_rfm
SAVE_PATH = 'models/segmentation/clustergan_model.pkl'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# === 1. Define Generator and Discriminator ===
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === 2. Train GAN ===
def train_clustergan(data, latent_dim=3, epochs=1000, batch_size=64, lr=0.0002):
    input_dim = data.shape[1]

    G = Generator(latent_dim, input_dim).to(device)
    D = Discriminator(input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    data_tensor = torch.FloatTensor(data.values).to(device)

    for epoch in range(epochs):
        idx = np.random.randint(0, data_tensor.shape[0], batch_size)
        real_data = data_tensor[idx]

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = G(z)

        D_real = D(real_data)
        D_fake = D(fake_data)

        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = G(z)
        D_fake = D(fake_data)

        loss_G = criterion(D_fake, real_labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    return G

# === 3. Generate cluster labels using float64 (to avoid API error) ===
def generate_cluster_labels(generator, data, latent_dim=3, n_clusters=4):
    with torch.no_grad():
        z = torch.randn(len(data), latent_dim).to(device)
        generated = generator(z).cpu().numpy()

    # ✅ Convert to float64
    generated = generated.astype(np.float64)

    # ✅ Train KMeans with float64 (important!)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(generated)

    return cluster_labels, kmeans

# === 4. Main ===
if __name__ == '__main__':
    os.makedirs("models/segmentation", exist_ok=True)

    print("📦 Loading and preprocessing data...")
    rfm_scaled = get_preprocessed_rfm()

    print("🧠 Training ClusterGAN...")
    G_model = train_clustergan(rfm_scaled)

    print("🔍 Generating clusters from latent space...")
    cluster_labels, kmeans_model = generate_cluster_labels(G_model, rfm_scaled)

    print("💾 Saving KMeans model for deployment...")
    joblib.dump(kmeans_model, SAVE_PATH)

    print("✅ ClusterGAN-based segmentation model saved to:", SAVE_PATH)

