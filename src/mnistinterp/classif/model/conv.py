import torch.nn as nn


class ConvMNISTClassifier(nn.Module):
    def __init__(self, embedding_size=512, kernel_size=3, dropout=0.2):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(1, embedding_size, kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(embedding_size, embedding_size, kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classify = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed(x)
        x = x.mean(dim=[2, 3])

        return self.classify(x)
