import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 60):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len//2).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_len, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        div_term = div_term.unsqueeze(1).expand(-1, d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, 0::2, :] = torch.sin(position * div_term)
        pe[0, 1::2, :] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
