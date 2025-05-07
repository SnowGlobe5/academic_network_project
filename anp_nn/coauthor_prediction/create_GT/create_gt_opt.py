import torch
import pandas as pd
import random
import json
import os
import time
import ast
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv
from torch_geometric.utils import coalesce
from tqdm import tqdm

from scipy.sparse import coo_matrix

# Carica gli embedding e i conteggi delle connessioni
BATCH_SIZE = 4096 
YEAR = 2019

infosphere_type = int(sys.argv[1])
infosphere_parameters = sys.argv[2]
only_new = False

# Set the random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
if not only_new:
    if infosphere_type == 1:
        author_embeddings_path = f'../anp_models/anp_embedding_generation_2025_1_5_False_-1_0.0_2025_04_15_18_25_52/author_embeddings.npy'
    else:
        author_embeddings_path = f'../anp_models/anp_embedding_generation_2025_1_5_False_-1_0.0_2025_04_15_18_25_52/author_embeddings_{infosphere_type}_{infosphere_parameters}.npy'
else:
    author_embeddings_path = f'../anp_models/anp_embedding_generation_2025_1_5_True_-1_0.0_2025_04_15_09_01_32/author_embeddings.npy'
print(f"Loading author embeddings from {author_embeddings_path}...")
author_embeddings = np.load(author_embeddings_path)
author_embeddings_tensor = torch.tensor(author_embeddings, dtype=torch.float32)
author_embeddings_dict = { "author": author_embeddings_tensor.to(DEVICE) }

author_edge_counts_path = f'../anp_data/processed/author_edge2020_count.pt'
max_author_degrees = torch.load(author_edge_counts_path).to(DEVICE)

# load co-author edges
co_author_infosphere_path = f'../anp_data/processed/co_author_infosphere/co_author_{infosphere_type}_{infosphere_parameters}_{YEAR}.pt'
if infosphere_type == 1:
    co_author_candidates = torch.load(co_author_infosphere_path, map_location=lambda storage, loc: storage)
else:
    co_author_candidates_top = torch.load(co_author_infosphere_path, map_location=lambda storage, loc: storage)
    co_author_history_path = f'../anp_data/processed/co_author_infosphere/co_author_history_{YEAR}.pt'
    co_author_candidates_history = torch.load(co_author_history_path, map_location=lambda storage, loc: storage)


DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *


# Path to the pre-trained model
if only_new:
    model_path = '../anp_models/anp_embedding_generation_2025_1_5_True_-1_0.0_2025_04_15_09_01_32/model.pt'
else:
    model_path = '../anp_models/anp_embedding_generation_2025_1_5_False_-1_0.0_2025_04_15_18_25_52/model.pt'

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Define model components
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                            num_heads)
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['author'][row], z_dict['author'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 1, 2, data.metadata())
        # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Initialize and load the model
model = torch.load(model_path, map_location=DEVICE)
model.eval()
num_authors = author_embeddings.shape[0]

# Function to calculate connection probability based on embeddings
def connection_probability(a, b):
    # Ensure a is a tensor of the same shape as b
    a_tensor = torch.full((b.shape[0],), a).to(DEVICE)  # Repeat a for the number of elements in b
    # Create a combined tensor for decoder input
    combined_tensor = torch.stack((a_tensor, b)).to(DEVICE)  # Concatenate a and b
    return model.decoder(author_embeddings_dict, combined_tensor)

# Looping through authors to expand the co-author set and generate connections
start_time = time.time()

if infosphere_type == 1:
    filtered_authors = [
        author_node for author_node in range(num_authors)
        if max_author_degrees[author_node] != 0 and co_author_candidates[author_node] != []
    ]
elif infosphere_type == 2:
    authors_df = pd.read_csv("../anp_data/processed/relevant_authors_2020.csv")
    filtered_authors = authors_df['author_id'].tolist()
else:
    filtered_authors = [
    author_node for author_node in range(num_authors)
    if max_author_degrees[author_node] != 0 
    and co_author_candidates_top[author_node] not in [None, []] 
    and (not isinstance(co_author_candidates_top[author_node], torch.Tensor) 
         or co_author_candidates_top[author_node].numel() > 0) 
    ]

current_author_degrees = torch.zeros(num_authors, device=DEVICE)

edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

probability_dict = {}

for j, author_node in enumerate(filtered_authors):
    if current_author_degrees[author_node] >= max_author_degrees[author_node]:
        continue

    if infosphere_type == 1:
        coauthor_candidates = co_author_candidates[author_node].to(DEVICE)
    elif infosphere_type == 2:
        combined_tensor = torch.cat((co_author_candidates_top, co_author_candidates_history[author_node]), dim=0)
        coauthor_candidates = torch.unique(combined_tensor, dim=0).to(DEVICE)
    else:
        combined_tensor = torch.cat((co_author_candidates_top[author_node], co_author_candidates_history[author_node]), dim=0)
        coauthor_candidates = torch.unique(combined_tensor, dim=0).to(DEVICE)

    valid_mask = (coauthor_candidates > author_node) & \
                 (current_author_degrees[coauthor_candidates] < max_author_degrees[coauthor_candidates])
    valid_candidates = coauthor_candidates[valid_mask]

    if valid_candidates.size(0) == 0:
        continue
    
    prob_mask = torch.tensor([probability_dict.get((author_node, candidate), 0) == 0 for candidate in valid_candidates])
    new_probs = connection_probability(author_node, valid_candidates[prob_mask]).clamp(min=0, max=1)
    
    for i, candidate in enumerate(valid_candidates[prob_mask]):
        probability_dict[(author_node, candidate.item())] = new_probs[i].item()

    connections = torch.tensor([probability_dict[(author_node, candidate.item())] for candidate in valid_candidates])
    
    connected_nodes = valid_candidates[connections > 0.5]

    sorted_indices = torch.argsort(connections[connections > 0.5], descending=True)
    connected_nodes = connected_nodes[sorted_indices]

    for coauthor_node in connected_nodes:
        edge_index = torch.cat((edge_index, torch.tensor([[author_node], [coauthor_node]], device=DEVICE)), dim=1)
        current_author_degrees[author_node] += 1
        current_author_degrees[coauthor_node] += 1

        if current_author_degrees[author_node] >= max_author_degrees[author_node]:
            break

    if (j - 1) % 1000 == 0:
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (j + 1)) * (len(filtered_authors) - j - 1)
        print(f"Processed {j + 1}/{len(filtered_authors)} authors. Estimated time remaining: {remaining_time / 60:.2f} minutes")


path = f'../anp_data/processed/gt_edge_index_{infosphere_type}_{infosphere_parameters}_{YEAR}_new_3.pt'
torch.save(edge_index, path)
path = f'../anp_data/processed/gt_probability_dict_{infosphere_type}_{infosphere_parameters}_{YEAR}_new_3.pt'
torch.save(probability_dict, path)


print("Matrices saved successfully.")

