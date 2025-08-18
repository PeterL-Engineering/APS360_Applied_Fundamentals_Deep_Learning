import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

class TweetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(TweetRNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.emb(x)                                     # Look-up the embeddings
        h0 = torch.zeros(1, x.size(0), self.hidden_size)    # Initial hidden state
        out, _ = self.rnn(x, h0)                            # Forward propagate RNN
        return self.fc(out[:, -1, :])                       # Classify last hidden state

model = TweetRNN(50, 64, 2)

def train(mode, train, val, n_epochs=5, lr=1e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mode.parameters(), lr = lr)

    for epoch in range(n_epochs):
        for tweets, labels in train:
            optimizer.zero_grad()
            pred = model(tweets)
            loss = criterion(pred, labels)
            loss.backward()
            loss.step()
