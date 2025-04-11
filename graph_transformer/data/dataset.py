import torch
from .graph import get_boardgame_graph
from setup import *

def separate_user_game_edges(data):
    """ Identify user->game edges versus game->game edges in a homogeneous Data object. Assumes that user nodes have
    indices [0, num_users) and game nodes have indices [num_users, num_users + num_games). The edges are undirected
    and stored as pairs (user->game, game->user), so we gather the forward edges only once. """
    num_users = data.num_users
    num_games = data.num_games
    # Parse data.edge_index to determine which edges are user->game.
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    src = edge_index[0]
    dst = edge_index[1]

    # Condition for user->game:
    # user node: 0 <= node < num_users
    # game node: num_users <= node < num_users + num_games
    is_ug_forward = (src < num_users) & (dst >= num_users) & (dst < (num_users + num_games))

    ug_forward_idx = torch.where(is_ug_forward)[0]

    ug_src = src[ug_forward_idx]
    ug_dst = dst[ug_forward_idx]
    ug_attr = edge_attr[ug_forward_idx]

    # Support edges are all other edges (e.g. game–game edges)
    is_support = ~is_ug_forward
    support_idx = torch.where(is_support)[0]

    support_src = src[support_idx]
    support_dst = dst[support_idx]
    support_attr = edge_attr[support_idx]

    return (ug_src, ug_dst, ug_attr), (support_src, support_dst, support_attr)


def random_split_user_game_edges(ug_src, ug_dst, ug_attr, val_ratio=0.1, test_ratio=0.1, seed=42):
    """ Split the user->game edges into train, validation, and test sets by random partition. Note that we do not
    split the support edges. """
    torch.manual_seed(seed)
    E = ug_src.size(0)
    perm = torch.randperm(E)
    val_size = int(val_ratio * E)
    test_size = int(test_ratio * E)

    val_idx = perm[:val_size]
    test_idx = perm[val_size:val_size + test_size]
    train_idx = perm[val_size + test_size:]

    return train_idx, val_idx, test_idx


def build_full_adjacency(data, ug_train, ug_val, ug_test, support):
    """ Assemble the full graph for message passing by combining: - Only the training user->game edges (ug_train) -
    All support edges (game->game) The validation and test user->game edges (ug_val and ug_test) are intentionally
    omitted to prevent label leakage during node embedding computation. """
    (ug_src_train, ug_dst_train, ug_attr_train) = ug_train
    (sup_src, sup_dst, sup_attr) = support
    edge_src = torch.cat([ug_src_train, sup_src], dim=0)
    edge_dst = torch.cat([ug_dst_train, sup_dst], dim=0)
    edge_attr = torch.cat([ug_attr_train, sup_attr], dim=0)

    new_edge_index = torch.stack([edge_src, edge_dst], dim=0)
    data.edge_index = new_edge_index
    data.edge_attr = edge_attr

    return data


def build_link_neighbor_loaders(data, ug_src, ug_dst, ug_attr, train_idx, val_idx, test_idx, num_neighbors=[10, 10], batch_size=1024):
    """ Create LinkNeighborLoader objects to sample subgraphs around the user->game edges for train, validation,
    and test splits. The user->game edges are used as link labels. """
    from torch_geometric.loader import LinkNeighborLoader
    train_src = ug_src[train_idx]
    train_dst = ug_dst[train_idx]
    train_attr = ug_attr[train_idx]

    val_src = ug_src[val_idx]
    val_dst = ug_dst[val_idx]
    val_attr = ug_attr[val_idx]

    test_src = ug_src[test_idx]
    test_dst = ug_dst[test_idx]
    test_attr = ug_attr[test_idx]

    train_edge_label_index = torch.stack([train_src, train_dst], dim=0)
    val_edge_label_index = torch.stack([val_src, val_dst], dim=0)
    test_edge_label_index = torch.stack([test_src, test_dst], dim=0)

    train_loader = LinkNeighborLoader(
        data,
        edge_label_index=train_edge_label_index,
        edge_label=train_attr,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = LinkNeighborLoader(
        data,
        edge_label_index=val_edge_label_index,
        edge_label=val_attr,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = LinkNeighborLoader(
        data,
        edge_label_index=test_edge_label_index,
        edge_label=test_attr,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def get_dataset_and_loaders_for_user_game_link_prediction( add_game_game_edges=False, add_shared_attribute_edges=False, similarity_threshold=0.75, top_k=10, val_ratio=0.1, test_ratio=0.1, num_neighbors=[10, 10], batch_size=1024 ):
    """ This function builds the dataset and constructs the loader objects for training, validation, and testing a
    user->game link prediction model. 1) Build the full graph using all user->game edges and game->game support
    edges. 2) Separate the user->game edges from the support edges. 3) Randomly split the user->game edges into
    train, val, and test sets. 4) Reassemble the graph for message passing using only the training edges plus support
    edges. 5) Construct LinkNeighborLoader objects for predicting the validation and test edges. """
    data = get_boardgame_graph(
        add_game_game_edges=add_game_game_edges,
        add_shared_attribute_edges=add_shared_attribute_edges,
        similarity_threshold=similarity_threshold,
        top_k_cont=top_k)
    (ug_src, ug_dst, ug_attr), (sup_src, sup_dst, sup_attr) = separate_user_game_edges(data)
    train_idx, val_idx, test_idx = random_split_user_game_edges(ug_src, ug_dst, ug_attr,
                                                                val_ratio=val_ratio, test_ratio=test_ratio,
                                                                seed=Config.SEED)

    ug_train = (ug_src[train_idx], ug_dst[train_idx], ug_attr[train_idx])
    ug_val = (ug_src[val_idx], ug_dst[val_idx], ug_attr[val_idx])
    ug_test = (ug_src[test_idx], ug_dst[test_idx], ug_attr[test_idx])
    support = (sup_src, sup_dst, sup_attr)

    data = build_full_adjacency(data, ug_train, ug_val, ug_test, support)

    train_loader, val_loader, test_loader = build_link_neighbor_loaders(
        data, ug_src, ug_dst, ug_attr, train_idx, val_idx, test_idx,
        num_neighbors=num_neighbors, batch_size=batch_size
    )

    return (data, train_loader, val_loader, test_loader,
            ug_src, ug_dst, ug_attr, train_idx, val_idx, test_idx)



