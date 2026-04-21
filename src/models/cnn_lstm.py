import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 14 * 14, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)

        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        return outputs


class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim)
        self.decoder = DecoderRNN(embed_dim, hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
