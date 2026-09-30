# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

class PositionalEncoding(nn.Module):
    # batch_first=True
    def __init__(self, d_emb, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2, dtype=torch.float) * (-log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_emb)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
