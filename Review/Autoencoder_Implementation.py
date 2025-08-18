import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        encoding_dim = 32
        self.encoder = nn.Linear(28 * 28, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, img):
        flattened = img.view(-1, 28 * 28)
        x = self.encoder(flattened)
        x = F.sigmoid(self.decoder(x)) # Sigmoid for scaling output from 0 to 1
        return x
    
    criterion = nn.MSELoss()

# Adding noise to input images

model = Autoencoder

nf = 0.4                                        # How much noise to add to images
noisy_img = img + nf * torch.randn(*img.shape)  # Add random noise to the input images
noisy_img = np.clip(noisy_img, 0., 1.)          # Clip the images to be between 0 and 1
outputs = model(noisy_img)                      # Compute predicted outputs using noisy_img
loss = criterion(outputs, img)                  # The target is the original image

# Example of a convolutional autoencoder

class Conv_Autoencoder(nn.Module):
    def __init__(self):
        super(Conv_Autoencoder).__init__()
        self.encoder == nn.Sequential (
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def embed(self, x):
        return self.encoder(x)
    
    def decode(self, e):
        return self.decode(e)