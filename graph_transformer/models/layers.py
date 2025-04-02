"""
This code is buggy ughhhh dimensional issues
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GraphAttentionLayer(MessagePassing):
    """
    A custom GAT-like layer built from scratch using PyG's MessagePassing.
    If you prefer, you can just use torch_geometric.nn.GATConv directly.
    """
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.6, concat=True):
        super().__init__(aggr='add')  # "Add" aggregation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        # Learnable linear projection for source/target nodes
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_dim))

        # Optional bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_dim))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_dim))

        # Init parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # Linear projection
        x_proj = self.lin(x)  # [num_nodes, heads*out_dim]
        # Reshape for multi-head: [num_nodes, heads, out_dim]
        x_proj = x_proj.view(-1, self.heads, self.out_dim).contiguous()

        # Let propagate automatically infer the size from x_proj
        out = self.propagate(edge_index, x=x_proj)

        # Reshape output based on concat setting
        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)  # average over heads

        out = out + self.bias
        return out

    def message(self, x_j, x_i, index):
        """
        x_j: [E, heads, out_dim] => neighbor embeddings
        x_i: [E, heads, out_dim] => center embeddings (for attention calc)
        index: target indices
        """
        # Compute attention coefficients
        # shape: [E, heads, out_dim]
        alpha = (x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1)
        # alpha: [E, heads]

        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, num_nodes=x_i.size(0))

        # Dropout on attention weights
        alpha = self.dropout(alpha)

        # Multiply neighbor features by attention
        out = x_j * alpha.unsqueeze(-1)  # [E, heads, out_dim]
        return out

    def update(self, inputs):
        # inputs: [num_nodes, heads, out_dim], aggregated by "add"
        return inputs


class PositionalEncoding(nn.Module):
    """
    Example sinusoidal positional encoding (like in classical Transformers).
    For graphs, you might do something more structural (Laplacian eigenmaps, etc.).
    We'll do a simple sinusoidal encoding of node indices for demonstration.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a [max_len, d_model] table of sine/cosine values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's not a learnable parameter but is moved to GPU, etc.
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        """
        x: [N, d_model], we'll add the positional embeddings based on node index or offset
        offset: which index to start from
        This is a toy example: we might do something else for graph nodes.
        """
        N = x.size(0)
        if N + offset > self.pe.size(0):
            raise ValueError("PositionalEncoding buffer not large enough.")

        x = x + self.pe[offset:offset+N, :]
        return x


class EdgeMLP(nn.Module):
    """
    A small MLP that takes user and game embeddings (concatenated or elementwise-multiplied)
    and predicts a rating.
    """
    def __init__(self, embed_dim, hidden_dim=64, out_dim=1, num_layers=2):
        super().__init__()
        layers = []
        in_dim = embed_dim * 2  # if we do user_emb concat game_emb
        for i in range(num_layers):
            outd = hidden_dim if i < num_layers-1 else out_dim
            fc = nn.Linear(in_dim, outd)
            layers.append(fc)
            if i < num_layers-1:
                layers.append(nn.ReLU())
            in_dim = outd
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_emb, game_emb):
        """
        user_emb: [batch_size, embed_dim]
        game_emb: [batch_size, embed_dim]
        returns: [batch_size, out_dim] => rating or logit
        """
        x = torch.cat([user_emb, game_emb], dim=-1)  # [batch_size, 2*embed_dim]
        return self.mlp(x)
