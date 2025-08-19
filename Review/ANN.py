import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

# Example 2-layer neural network

class NN(nn.module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 30)
        self.drop = nn.Dropout(p=0.3)
        self.layer2 =  nn.Linear(30, 1)

    def foward(self, img):
        flattened = img.view(-1, 28 * 28)
        activation1 = self.drop(self.layer1(flattened))
        activation2 = F.relu(activation1)
        activation2 - self.drop(self.layer2(activation1))
        return activation2
    
model = NN()

# Example training code for binary classification problem

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

for (image, label) in mnist_train:
    # Ground truth: is the digit less than 3?
    actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)

    out - model(img_to_tensor(image))   # Make prediction
    loss = criterion(out, actual)       # Calculate loss
    loss.backward()                     # Obtain gradients
    optimizer.step()                    # Update parameters
    optimizer.zero_grad()               # Clean up gradients

# Computing the error and accuracy on a training set
error = 0
for (image, label) in mnist_train:
    prob - torch.sigmoid(model(img_to_tensor(image)))
    if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >- 3):
        error += 1
print("Training Error Rate:", error/len(mnist_train))
print("Training Accuracy:", 1 - error/len(mnist_train))

# Example Multi-class ANN Architecture

class MNISTCLassifier(nn.Module):
    def __init__(self):
        super(MNISTCLassifier, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 50)
        self.bn = nn.BatchNorm1d(50)
        self.layer2 = nn.Linear(50, 20)
        self.ln = nn.LayerNorm(20)
        self.layer3 = nn.Linear(20, 10) # One output neuron for each of the ten digits

    def forward(self, img):
        flattened = img.view(-1, 28 * 28)
        activation1 = self.bn(F.relu(self.layer1(flattened)))
        activation2 = self.ln(F.relu(self.layer2(activation1)))
        logits = self.layer3(activation2)
        probs = F.softmax(logits, dim=1)
        return probs
        
model = MNISTCLassifier()