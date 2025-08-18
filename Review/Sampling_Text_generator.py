import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(1)

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(TextGenerator, self).__init__()
        self.ident = torch.eye(vocab_size)                              # Identity matrix for generating 1-hot vectors
        self.rnn = nn.GRU(vocab_size, hidden_size, batch_first=True)    # Recurrent neural network
        self.decoder = nn.Linear(hidden_size, vocab_size)               # A FC layer outputting a distribution over the next token

    def forward(self, inp, hidden=None):
        inp = self.ident[inp]                   # Generate 1-hot vectors of input
        output, hidden = self.rnn(inp, hidden)  # Get the next output and hidden state
        output = self.decoder(output)           # Predict distribution over next tokens
        return output, hidden
    
    def train(model, data, batch_size=1, num_epochs=1, lr=0.01):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion == nn.CrossEntropyLoss()
        data_iter = torchtext.legacy.data.BucketIterator(data,
                                                         batch_size=batch_size,
                                                         sort_key= lambda x: len(x.text),
                                                         sort_within_batch=True)
        
        for _ in range(num_epochs):
            Avg_loss = 0
            for (tweet, lengths), label in data_iter:
                target = tweet[:, 1:]
                inp = tweet[:, :-1]
                optimizer.zero_grad()
                loss, _ = model(inp)
                loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
                loss.backward()
                optimizer.step()

    def sample(sample, max_len=100, temperature=0.8):
        generated_sequence = ''
        inp = torch.Tensor([vocab_stoi['<BOS>']]).long()
        for p in range(max_len):
            output, hidden = model(inp.unsqueeze(0), hidden)

            # Sample from the model as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = int(torch.multinomial(output_dist, 1)[0])

            # Add predicted character to string and use as next input
            predicted_char = vocab_itos[top_i]
            if predicted_char == '<EOS>':
                break
            generated_sequence += predicted_char
            inp = torch.Tensor([top_i]).long()
        return generated_sequence