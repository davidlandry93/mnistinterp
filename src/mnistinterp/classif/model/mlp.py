import torch.nn as nn

from einops import rearrange


class MNISTClassifier(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.2):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(32 * 32, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
        )

        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = rearrange(x, "b w h -> b (w h)")

        return self.classify(self.embed(x))
