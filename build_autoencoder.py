import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define a dummy dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(1, 640, 640)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image tensor (normalized between 0 and 1)
        return torch.rand(*self.image_size)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize dataset and dataloader
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize model, optimizer, and loss function
autoencoder = Autoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(autoencoder.state_dict(), "autoencoder.pth")
print("Model saved!")

# Reload the model
autoencoder_reload = Autoencoder()
autoencoder_reload.load_state_dict(torch.load("autoencoder.pth"))
print("Model reloaded!")

# Verify model reload
sample_input = torch.rand(1, 1, 640, 640)  # A single image
output = autoencoder_reload(sample_input)
print("Input shape:", sample_input.shape)
print("Output shape:", output.shape)

# Mean Absolute Error (MAE): This calculates the average absolute difference between the input and output.

reconstruction_error = torch.mean(torch.abs(output_image - input_image))
print("Reconstruction Error (MAE):", reconstruction_error.item())
