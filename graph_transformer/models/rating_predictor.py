import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from setup import *

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class GraphTransformer(nn.Module):
    """
    A stack of GATConv layers to produce node embeddings, with LayerNorm and residual connections.
    Now includes DropEdge for regularization.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=2, dropout=0.3, edge_dropout=Config.EDGE_DROPOUT):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()  # layer normalization layers

        # 1st layer
        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        self.lns.append(nn.LayerNorm(hidden_dim * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
            )
            self.lns.append(nn.LayerNorm(hidden_dim * num_heads))

        # Final layer (with averaged heads, so concat=False)
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_dim * num_heads, out_dim, heads=num_heads, dropout=dropout, concat=False)
            )
            self.lns.append(nn.LayerNorm(out_dim))

        self.act = nn.ReLU()
        self.apply(init_weights)

    def drop_edge(self, edge_index, drop_prob):
        """Randomly drop edges with probability drop_prob."""
        device = edge_index.device
        n_edges = edge_index.size(1)
        mask = torch.rand(n_edges, device=device) >= drop_prob
        return edge_index[:, mask]

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_in = x  # Save input for potential residual connection
            # Apply edge dropout separately in each layer during training
            if self.training and self.edge_dropout > 0:
                effective_edge_index = self.drop_edge(edge_index, self.edge_dropout)
            else:
                effective_edge_index = edge_index

            x = conv(x, effective_edge_index)
            x = self.lns[i](x)
            if i < self.num_layers - 1:
                # Add residual connection when dimensions match
                if x.shape == x_in.shape:
                    x = x + x_in
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class EdgeMLP(nn.Module):
    """
    An MLP that combines user and game embeddings to predict a rating,
    using LayerNorm and a lower dropout rate for improved gradient flow.
    """
    def __init__(self, embed_dim, hidden_dim=64, out_dim=1, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        layers = []
        self.lns = nn.ModuleList()  # layer normalization layers for hidden layers
        in_dim = embed_dim * 2  # because user and game embeddings are concatenated
        self.pre_mlp_norm = nn.LayerNorm(embed_dim * 2)
        for layer_idx in range(num_layers):
            layer_out_dim = hidden_dim if layer_idx < num_layers - 1 else out_dim
            linear_layer = nn.Linear(in_dim, layer_out_dim)
            layers.append(linear_layer)
            if layer_idx < num_layers - 1:
                self.lns.append(nn.LayerNorm(layer_out_dim))
            in_dim = layer_out_dim

        self.linears = nn.ModuleList(layers)
        # An extra final adjustment layer
        self.output_layer = nn.Linear(out_dim, out_dim)
        self.apply(init_weights)

    def forward(self, user_emb, game_emb):
        x = torch.cat([user_emb, game_emb], dim=-1)
        x = self.pre_mlp_norm(x)
        bn_count = 0
        for layer_idx, linear in enumerate(self.linears):
            x = linear(x)
            if layer_idx < self.num_layers - 1:
                x = self.lns[bn_count](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                bn_count += 1
        rating_raw = self.output_layer(x)
        return rating_raw

class RatingPredictor(nn.Module):
    """
    Full rating prediction model that:
      1) Uses a GraphTransformer (with LayerNorm, residual connections, and DropEdge) to compute node embeddings.
      2) Adds trainable embeddings for users and games.
      3) Uses an EdgeMLP to predict ratings from the combined embeddings.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=2, dropout=0.3,
                 rating_hidden=64, rating_out=1, rating_layers=2,
                 num_users=0, num_games=0):
        super().__init__()
        self.gnn = GraphTransformer(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dropout=Config.EDGE_DROPOUT  # pass the dropout value from config
        )
        self.edge_mlp = EdgeMLP(
            embed_dim=out_dim,
            hidden_dim=rating_hidden,
            out_dim=rating_out,
            num_layers=rating_layers,
            dropout=dropout
        )
        self.num_users = num_users
        self.num_games = num_games

        # Trainable embeddings for user and game IDs
        if num_users > 0:
            self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=out_dim)
        else:
            self.user_embedding = None
        if num_games > 0:
            self.game_embedding = nn.Embedding(num_embeddings=num_games, embedding_dim=out_dim)
        else:
            self.game_embedding = None

        self.apply(init_weights)

    def forward(self, x, edge_index, edge_label_index, n_id=None):
        # Compute node embeddings via the GraphTransformer
        node_emb = self.gnn(x, edge_index)

        # For neighbor sampling: add ID embeddings to the corresponding subset of nodes.
        if n_id is not None:
            if self.user_embedding is not None:
                user_mask = n_id < self.num_users
                if user_mask.any():
                    node_emb[user_mask] += self.user_embedding(n_id[user_mask])
            if self.game_embedding is not None:
                game_mask = n_id >= self.num_users
                if game_mask.any():
                    game_ids = n_id[game_mask] - self.num_users
                    node_emb[game_mask] += self.game_embedding(game_ids)
        else:
            # Full-batch approach: assign embeddings for all users and games.
            if self.user_embedding is not None:
                user_indices = torch.arange(self.num_users, device=node_emb.device)
                node_emb[user_indices] += self.user_embedding(user_indices)
            if self.game_embedding is not None:
                game_indices = torch.arange(self.num_games, device=node_emb.device)
                node_emb[self.num_users + game_indices] += self.game_embedding(game_indices)

        # Gather embeddings for user-game pairs from edge_label_index.
        src = edge_label_index[0]
        dst = edge_label_index[1]
        user_emb_batch = node_emb[src]
        game_emb_batch = node_emb[dst]

        # Predict rating using the EdgeMLP.
        rating_pred = self.edge_mlp(user_emb_batch, game_emb_batch)
        return rating_pred
