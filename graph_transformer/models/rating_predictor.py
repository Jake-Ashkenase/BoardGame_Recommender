import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphTransformer(nn.Module):
    """
    A stack of GATConv layers to produce node embeddings.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # 1st layer: from input to hidden
        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        )
        # middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
            )
        # final layer: output dimension with averaging heads
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_dim * num_heads, out_dim, heads=num_heads, dropout=dropout, concat=False)
            )
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class RatingPredictor(nn.Module):
    """
    Full model for user->game rating prediction.
    1) A GraphTransformer (stack of GATConv layers) to compute node embeddings.
    2) An EdgeMLP that combines user and game embeddings to predict a rating.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=2, dropout=0.5,
                 rating_hidden=64, rating_out=1, rating_layers=2):
        super().__init__()
        self.gnn = GraphTransformer(in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout)
        self.edge_mlp = EdgeMLP(embed_dim=out_dim,
                                hidden_dim=rating_hidden,
                                out_dim=rating_out,
                                num_layers=rating_layers)

    def forward(self, x, edge_index, edge_label_index):
        # get node embeddings using the GNN
        node_emb = self.gnn(x, edge_index)  # [num_nodes, out_dim]
        # gather embeddings for user->game edge endpoints
        src = edge_label_index[0]
        dst = edge_label_index[1]
        user_emb = node_emb[src]
        game_emb = node_emb[dst]
        # predict rating
        rating_pred = self.edge_mlp(user_emb, game_emb)  # [batch_size, 1]
        return rating_pred

class EdgeMLP(nn.Module):
    """
    A small MLP that takes concatenated user and game embeddings and predicts a rating.
    """
    def __init__(self, embed_dim, hidden_dim=64, out_dim=1, num_layers=2):
        super().__init__()
        layers = []
        in_dim = embed_dim * 2  # assuming user and game embeddings are the same dimension
        for i in range(num_layers):
            outd = hidden_dim if i < num_layers-1 else out_dim
            layers.append(nn.Linear(in_dim, outd))
            if i < num_layers-1:
                layers.append(nn.ReLU())
            in_dim = outd
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_emb, game_emb):
        x = torch.cat([user_emb, game_emb], dim=-1)
        return self.mlp(x)
