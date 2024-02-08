import torch.nn as nn

from attention import MultiHeadAttention

dropout = 0.2


class FeedForward(nn.Module):
    """Linear layer followed by ReLU"""

    def __init__(self, n_embd):
        super().__init__()
        self.multiplier = 4
        self.net = nn.Sequential(
            nn.Linear(n_embd, self.multiplier * n_embd),
            nn.ReLU(),
            nn.Linear(self.multiplier * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication and computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
