import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out.view(x.size(0))
    
    def train_discriminator(discriminator, generator, images):
        batch_size = images.size(0)
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        inputs = torch.cat([images, fake_images])
        labels = torch.cat([torch.zeros(batch_size),    # Real
                            torch.ones(batch_size)])    # Fake
        outputs = discriminator(inputs)
        loss = criterion(outputs, labels)
        return outputs, loss

class Generator(nn.Module):
    def __initi__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x).view(x.size(0), 1, 28, 28)
        return out.view(x.size(0))
    
    def train_generator(discriminator, generator, batch_size):
        batch_size = images.size(0)
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        # Only looks at fake outputs
        # Gets rewarded if we fool the discriminator
        labels = torch.zeros(batch_size)
        loss = criterion(outputs, labels)
        return fake_images, loss