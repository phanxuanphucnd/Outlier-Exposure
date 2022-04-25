import torch.nn as nn


class ClfGRU(nn.Module):
    def __init__(self, num_classes, TEXT):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(
            input_size=50,
            hidden_size=128,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.linear = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]
        logits = self.linear(hidden)
        return logits
