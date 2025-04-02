import os
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data
from setup import *

try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None


class BoardGameGraphBuilder:
    """
    More efficient version that uses:
      - Inverted index for binary columns
      - Approx nearest neighbor for continuous columns
      - Chunk-based reading for user->game edges
    """

    def __init__(
        self,
        binary_start_index=512,
        top_k_cont=10,
        star_link_for_binary=True,
        large_set_threshold=5000
    ):
        """
        :param binary_start_index: columns < this are continuous, >= are binary
        :param top_k_cont: number of nearest neighbors to fetch for each game in continuous space
        :param star_link_for_binary: if True, connect each set of games in a star instead of a clique
        :param large_set_threshold: skip or reduce sets bigger than this for binary columns to avoid huge edges
        """
        self.binary_start_index = binary_start_index
        self.top_k_cont = top_k_cont
        self.star_link_for_binary = star_link_for_binary
        self.large_set_threshold = large_set_threshold

        self.games_df = None
        self.ratings_path = Config.USER_RATINGS_FILE
        self.user2id = {}
        self.game2id = {}

        self.num_users = 0
        self.num_games = 0

        # store edges in a memory-efficient way:
        self.edge_src = []
        self.edge_dst = []
        self.edge_attr = []

    def load_game_data(self):
        # read the games csv
        if not os.path.exists(Config.OVERALL_GAMES_FILE):
            raise FileNotFoundError("Game data file not found.")
        self.games_df = pd.read_csv(Config.OVERALL_GAMES_FILE)
        for col in self.games_df.columns:
            if col not in ["BGGId", "Name"]:
                # Convert to numeric if possible, otherwise NaN
                self.games_df[col] = pd.to_numeric(self.games_df[col], errors='coerce')
        if "BGGId" not in self.games_df.columns:
            raise ValueError("Missing BGGId in games CSV.")

        self.games_df.dropna(inplace=True)
        self.games_df.reset_index(drop=True, inplace=True)
        # build game->id map
        bgg_ids = self.games_df["BGGId"].unique()
        self.game2id = {bgg_id: idx for idx, bgg_id in enumerate(bgg_ids)}
        self.num_games = len(self.game2id)

    def build_user_map_in_chunks(self, chunksize=100000):
        user_set = set()
        for chunk in pd.read_csv(self.ratings_path, chunksize=chunksize):
            # assume chunk has columns [Username, BGGId, rating]
            chunk.dropna(subset=["Username", "BGGId", "Rating"], inplace=True)
            user_set.update(chunk["Username"].unique())

        self.user2id = {u: i for i, u in enumerate(sorted(user_set))}
        self.num_users = len(self.user2id)
        user_set.clear()  # free memory

    def build_user_game_edges_in_chunks(self, chunksize=100000):
        """
        Stream through user_ratings.csv, building edges from user->game in memory-friendly chunks.
        """
        for chunk in pd.read_csv(self.ratings_path, chunksize=chunksize):
            chunk.dropna(subset=["Username", "BGGId", "Rating"], inplace=True)
            for _, row in chunk.iterrows():
                user = row["Username"]
                bggid = row["BGGId"]
                rating = row["Rating"]
                if (user not in self.user2id) or (bggid not in self.game2id):
                    continue
                u_idx = self.user2id[user]
                g_idx = self.game2id[bggid]
                g_node_idx = self.num_users + g_idx

                # forward edge
                self.edge_src.append(u_idx)
                self.edge_dst.append(g_node_idx)
                self.edge_attr.append(rating)

                # reverse edge
                self.edge_src.append(g_node_idx)
                self.edge_dst.append(u_idx)
                self.edge_attr.append(rating)

    def build_continuous_knn_edges(self, X_cont):
        """
        Uses approximate nearest neighbor (Annoy) to link each game to top_k_cont neighbors
        in the continuous columns. Bypasses O(N^2).
        """
        if AnnoyIndex is None:
            print("Annoy not installed. Skipping continuous KNN edges.")
            return

        n_games, dim = X_cont.shape
        print(f"Building Annoy index for {n_games} games, dimension={dim} ...")
        index = AnnoyIndex(dim, metric='euclidean')
        for i in range(n_games):
            index.add_item(i, X_cont[i].tolist())
        index.build(10)  # number of trees

        print(f"Querying top-{self.top_k_cont} neighbors for each game ...")
        for i in range(n_games):
            neighbors = index.get_nns_by_item(i, self.top_k_cont + 1)  # +1 to skip self
            # skip i from the results
            neighbors = [n for n in neighbors if n != i]
            for nbr in neighbors:
                src = self.num_users + i
                dst = self.num_users + nbr
                self.edge_src.append(src)
                self.edge_dst.append(dst)
                self.edge_attr.append(0.0)  # no rating
                # undirected => add reverse
                self.edge_src.append(dst)
                self.edge_dst.append(src)
                self.edge_attr.append(0.0)

    def build_binary_inverted_index_edges(self, X_bin):
        """
        For each binary column, gather the set of game IDs that have '1'.
        If the set size <= large_set_threshold, connect them (star or clique).
        If it is bigger, either skip or partially connect.
        """
        n_games, n_bin_cols = X_bin.shape
        # For each column, build a list of game indices that have 1
        for col_idx in range(n_bin_cols):
            ones = np.where(X_bin[:, col_idx] == 1)[0]  # game indices in [0..n_games-1]
            size = len(ones)
            if size < 2:
                continue
            if size > self.large_set_threshold:
                # skip to avoid a huge edge blow-up. plus attributes with sooo many in common prob aren't useful
                continue

            if self.star_link_for_binary:
                # star approach: pick the first game as hub
                hub = ones[0]
                for j in range(1, size):
                    game_j = ones[j]
                    # absolute IDs
                    src = self.num_users + hub
                    dst = self.num_users + game_j
                    self.edge_src.append(src)
                    self.edge_dst.append(dst)
                    self.edge_attr.append(0.0)
                    # reverse
                    self.edge_src.append(dst)
                    self.edge_dst.append(src)
                    self.edge_attr.append(0.0)
            else:
                # clique approach (prob not used due to compute blow up)
                # if size is large, this can be huge: size*(size-1)/2 edges
                for i in range(size):
                    for j in range(i+1, size):
                        game_i = ones[i]
                        game_j = ones[j]
                        src = self.num_users + game_i
                        dst = self.num_users + game_j
                        self.edge_src.append(src)
                        self.edge_dst.append(dst)
                        self.edge_attr.append(0.0)
                        # reverse
                        self.edge_src.append(dst)
                        self.edge_dst.append(src)
                        self.edge_attr.append(0.0)

    def build_graph(self, add_game_game_edges=False, add_shared_attribute_edges=False):
        # load game data
        self.load_game_data()

        # build user mapping in a streaming manner
        self.build_user_map_in_chunks()

        # build user->game edges (millions of ratings) in streaming manner
        self.build_user_game_edges_in_chunks()

        # now have self.edge_src/dst/attr for the user->game edges.

        total_nodes = self.num_users + self.num_games
        # build or gather the node features
        # ignore BGGId and Name bc they aren't features
        used_cols = [c for c in self.games_df.columns if c not in ("Name", "BGGId")]
        used_cols.sort()
        D = len(used_cols)

        X = torch.zeros((total_nodes, D), dtype=torch.float)

        # fill game rows
        bggid2row = {row["BGGId"]: row for _, row in self.games_df.iterrows()}
        for bggid, g_idx in self.game2id.items():
            node_idx = self.num_users + g_idx
            row = bggid2row[bggid]
            feats = [row[c] for c in used_cols]
            X[node_idx] = torch.tensor(feats, dtype=torch.float)


        # build game->game edges from continuous columns using ANN
        if add_game_game_edges and self.top_k_cont > 0 and D > self.binary_start_index:
            # separate out the continuous portion: columns [0 .. binary_start_index-1]
            X_cont = X[self.num_users:, :self.binary_start_index].numpy()
            self.build_continuous_knn_edges(X_cont)

        # build game->game edges from binary columns using an inverted index
        if add_shared_attribute_edges and D > self.binary_start_index:
            X_bin = X[self.num_users:, self.binary_start_index:].numpy()
            self.build_binary_inverted_index_edges(X_bin)

        # convert edges to torch Tensors
        edge_index = torch.tensor([self.edge_src, self.edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)

        # coalesce to remove duplicates (if edges have been built muultiple times btwn the same nodes, take max edge
        # value)
        from torch_geometric.utils import coalesce
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=total_nodes, reduce='max')

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
        data.num_users = self.num_users
        data.num_games = self.num_games
        data.user2id = self.user2id
        data.game2id = self.game2id
        data.feature_cols = used_cols

        return data


def get_boardgame_graph(
    binary_start_index=512,
    top_k_cont=10,
    similarity_threshold=0.75,
    star_link_for_binary=True,
    large_set_threshold=5000,
    add_game_game_edges=False,
    add_shared_attribute_edges=False
):
    """
    Usage:
      data = get_boardgame_graph()

    Args:
      binary_start_index: columns < this are continuous, >= are binary
      top_k_cont: how many nearest neighbors to link each game with in continuous space
      star_link_for_binary: if True, star linking sets for each binary column instead of clique
      large_set_threshold: skip sets bigger than this for binary columns
    """
    builder = BoardGameGraphBuilder(
        binary_start_index=binary_start_index,
        top_k_cont=top_k_cont,
        star_link_for_binary=star_link_for_binary,
        large_set_threshold=large_set_threshold
    )
    return builder.build_graph(
        add_game_game_edges=add_game_game_edges,
        add_shared_attribute_edges=add_shared_attribute_edges
    )
