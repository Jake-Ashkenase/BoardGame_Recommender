import torch
#from torch_geometric.loader import LinkNeighborLoader
from .graph import get_boardgame_graph


def separate_user_game_edges(data):
    """
    Identify user->game edges vs game->game edges in a homogeneous Data object.
    Can check if 'src' < data.num_users <= 'dst', which implies user->game.

    Edges are undirected and stored as pairs (u->g, g->u).
    Gather the user->game pairs once and skip duplicates.
    """
    num_users = data.num_users
    num_games = data.num_games

    # parse data.edge_index to figure out which edges are user->game
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    src = edge_index[0]
    dst = edge_index[1]

    # Condition for user->game:
    # user node: 0 <= user < num_users
    # game node: num_users <= game < num_users + num_games
    # edges are repeated for undirected (u->g, g->u) but don't want to double count

    is_ug_forward = (src < num_users) & (dst >= num_users) & (dst < (num_users + num_games))

    # gather these edges:
    ug_forward_idx = torch.where(is_ug_forward)[0]

    # user->game edges in the final adjacency might appear again in reverse (g->u)
    # the forward edges are the unique UG edges for splitting

    ug_src = src[ug_forward_idx]
    ug_dst = dst[ug_forward_idx]
    ug_attr = edge_attr[ug_forward_idx]

    # game->game edges are all edges connecting two nodes >= num_users
    # mask for everything that is not a forward UG edge and keep them as support edges
    is_support = ~is_ug_forward
    support_idx = torch.where(is_support)[0]

    support_src = src[support_idx]
    support_dst = dst[support_idx]
    support_attr = edge_attr[support_idx]

    return (ug_src, ug_dst, ug_attr), (support_src, support_dst, support_attr)


def random_split_user_game_edges(ug_src, ug_dst, ug_attr, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split the user->game edges into train/val/test sets by random partition.
    do not split the support edges. We'll keep them all in adjacency at all times.
    """
    torch.manual_seed(seed)
    E = ug_src.size(0)  # number of UG edges
    perm = torch.randperm(E)

    val_size = int(val_ratio * E)
    test_size = int(test_ratio * E)

    val_idx = perm[:val_size]
    test_idx = perm[val_size: val_size + test_size]
    train_idx = perm[val_size + test_size:]

    return train_idx, val_idx, test_idx


def build_full_adjacency(data, ug_train, ug_val, ug_test, support):
    """
    Combine:
      - all user->game edges (train, val, test)
      - all support edges (game->game)
    into data.edge_index, data.edge_attr.

    This means data.edge_index will contain everything
    won't train on the val/test edges. We'll only pass train edges to the LinkNeighborLoader.
    """
    # unpack
    (ug_src_train, ug_dst_train, ug_attr_train) = ug_train
    (ug_src_val, ug_dst_val, ug_attr_val) = ug_val
    (ug_src_test, ug_dst_test, ug_attr_test) = ug_test

    (sup_src, sup_dst, sup_attr) = support

    # concat
    edge_src = torch.cat([ug_src_train, ug_src_val, ug_src_test, sup_src], dim=0)
    edge_dst = torch.cat([ug_dst_train, ug_dst_val, ug_dst_test, sup_dst], dim=0)
    edge_attr = torch.cat([ug_attr_train, ug_attr_val, ug_attr_test, sup_attr], dim=0)

    # store in data
    new_edge_index = torch.stack([edge_src, edge_dst], dim=0)
    data.edge_index = new_edge_index
    data.edge_attr = edge_attr

    return data


def build_link_neighbor_loaders(data,
                                ug_src, ug_dst, ug_attr,
                                train_idx, val_idx, test_idx,
                                num_neighbors=[10, 10],
                                batch_size=1024):
    """
    Create LinkNeighborLoader objects that sample subgraphs around the
    user->game edges in train, val, test splits. pass in only UG edges
    for link_label purposes.
    """
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

    # stack the edge label indices into a tensor of shape [2, num_edges]
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


def get_dataset_and_loaders_for_user_game_link_prediction(
        add_game_game_edges=False,
        add_shared_attribute_edges=False,
        similarity_threshold=0.75,
        top_k=10,
        val_ratio=0.1,
        test_ratio=0.1,
        num_neighbors=[10, 10],
        batch_size=1024
):
    """
      1) build the full graph with user->game edges (having ratings)
         and  game->game edges as support (no rating).
      2) separate out user->game edges from the adjacency.
      3) splits user->game edges into train/val/test.
      4) reassembles the adjacency so that all edges remain, but only user->game train, val, test
         edges are used for link labeling. The optional game->game edges are support
      5) Builds LinkNeighborLoaders for train/val/test user->game edges.
    """
    # build the full graph
    data = get_boardgame_graph(
        add_game_game_edges=add_game_game_edges,
        add_shared_attribute_edges=add_shared_attribute_edges,
        similarity_threshold=similarity_threshold,
        top_k_cont=top_k
    )
    # data.edge_index: contains user->game edges (with rating) and game->game edges (attr=0)

    # separate user->game edges from support edges
    (ug_src, ug_dst, ug_attr), (sup_src, sup_dst, sup_attr) = separate_user_game_edges(data)

    # split user->game edges
    train_idx, val_idx, test_idx = random_split_user_game_edges(
        ug_src, ug_dst, ug_attr,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # reassemble the adjacency so data.edge_index has everything but we only use the splitted UG edges for label
    ug_train = (ug_src[train_idx], ug_dst[train_idx], ug_attr[train_idx])
    ug_val = (ug_src[val_idx], ug_dst[val_idx], ug_attr[val_idx])
    ug_test = (ug_src[test_idx], ug_dst[test_idx], ug_attr[test_idx])
    support = (sup_src, sup_dst, sup_attr)

    data = build_full_adjacency(data, ug_train, ug_val, ug_test, support)

    # build the LinkNeighborLoaders for only the user->game edges
    train_loader, val_loader, test_loader = build_link_neighbor_loaders(
        data, ug_src, ug_dst, ug_attr,
        train_idx, val_idx, test_idx,
        num_neighbors=num_neighbors,
        batch_size=batch_size
    )

    return data, train_loader, val_loader, test_loader
