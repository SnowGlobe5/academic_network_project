import torch
import pandas as pd
import random
import json
import os
import time
import ast
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
infosphere_type = 4
infosphere_parameters = 1
only_new = False
edge_number = 50
drop_percentage = 0.0
BATCH_SIZE = 4096  # Reduced batch size for memory optimization
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
author_embeddings_path = f'../anp_data/processed/embeddings/author_embeddings_{YEAR}_{infosphere_type}_{infosphere_parameters}.pt'
author_embeddings = torch.load(author_embeddings_path)
author_embeddings_dict = { "author": author_embeddings.to(DEVICE) }

author_edge_counts_path = f'../anp_data/processed/author_edge2020_count.pt'
max_author_degrees = torch.load(author_edge_counts_path).to(DEVICE)

# load co-author edges
co_author_infosphere_path = f'../anp_data/processed/co_author_infosphere/co_author_{infosphere_type}_{infosphere_parameters}_{YEAR}.pt'
co_author_candidates_top = torch.load(co_author_infosphere_path, map_location=lambda storage, loc: storage)

co_author_history_path = f'../anp_data/processed/co_author_infosphere/co_author_history_{YEAR}.pt'
co_author_candidates_history = torch.load(co_author_history_path, map_location=lambda storage, loc: storage)

DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *


# Path to the pre-trained model
model_path = '../anp_models_2_11/anp_link_prediction_co_author_hgt_embedding_faster_1_5_False_-1_0.0_2024_10_25_15_55_26/model.pt'


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere data if it was used in training
if infosphere_type != 0:
    if infosphere_type == 1:
        fold = [0, 1, 2, 3, 4]
        fold_string = '_'.join(map(str, fold))
        name_infosphere = f"{infosphere_parameters}_infosphere_{fold_string}_{YEAR}_noisy.pt"

        # Load infosphere
        if os.path.exists(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}"):
            infosphere_edges = torch.load(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}", map_location=DEVICE)
            
            # Drop edges for each type of relationship
            cites_edges = drop_edges(infosphere_edges[CITES], drop_percentage)
            writes_edges = drop_edges(infosphere_edges[WRITES], drop_percentage)
            about_edges = drop_edges(infosphere_edges[ABOUT], drop_percentage)
    
            data['paper', 'infosphere_cites', 'paper'].edge_index = coalesce(cites_edges)
            data['paper', 'infosphere_cites', 'paper'].edge_label = None
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(writes_edges)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            data['paper', 'infosphere_about', 'topic'].edge_index = coalesce(about_edges)
            data['paper', 'infosphere_about', 'topic'].edge_label = None
        else:
            raise Exception(f"{name_infosphere} not found!")
        
    elif infosphere_type == 2:
        infosphere_edge = create_infosphere_top_papers_edge_index(data, int(infosphere_parameters), YEAR)
        data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere_writes', 'paper'].edge_label = None

    elif infosphere_type == 3:
        arg_list = [5,2]
        if os.path.exists(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt"):
            print("Infosphere 3 edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Generating infosphere 3 edge...")
            infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            
    elif infosphere_type == 4:
        if os.path.exists(f"{ROOT}/processed/rec_edge_10_NAIS.pt"):
            print("Rec edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_NAIS.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()
    
    elif infosphere_type == 5:
        if os.path.exists(f"{ROOT}/processed/rec_edge_10_LightGCN.pt"):
            print("Rec edge found!")
            data['author', 'infosphere_writes', 'paper'].edge_index = torch.load(f"{ROOT}/processed/rec_edge_10_LightGCN.pt", map_location=DEVICE)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
        else:
            print("Error: Rec edge not found!")
            exit()


data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

# Define model components (same as training)
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
        self.decoder = EdgeDecoder(hidden_channels)
        self.embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32)
        self.embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Initialize and load the model
model = torch.load(model_path, map_location=DEVICE)
model.eval()

# Parametri
num_authors = author_embeddings.shape[0]

# Funzione per calcolare la probabilità di connessione basata sugli embedding
def connection_probability(a, b):
    # Ensure a is a tensor of the same shape as b
    a_tensor = torch.full((b.shape[0],), a).to(DEVICE)  # Repeat a for the number of elements in b
    # Create a combined tensor for decoder input
    combined_tensor = torch.stack((a_tensor, b)).to(DEVICE)  # Concatenate a and b
    return model.decoder(author_embeddings_dict, combined_tensor)


# Loop sugli autori per espandere il set di co-autori e generare connessioni
# Ciclo sugli autori
start_time = time.time()

filtered_authors = [
    author_node for author_node in range(num_authors)
    if max_author_degrees[author_node] != 0 
    and co_author_candidates_top[author_node] not in [None, []]  # Controlla che non sia None o lista vuota
    and (not isinstance(co_author_candidates_top[author_node], torch.Tensor) 
         or co_author_candidates_top[author_node].numel() > 0)  # Se è un tensore, controlla che non sia vuoto
]

current_author_degrees = torch.zeros(num_authors, device=DEVICE)

# Rappresentazione dell'edge index
edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

# Matrice delle probabilità sparsa
probability_dict = {}

for j, author_node in enumerate(filtered_authors):
    if current_author_degrees[author_node] >= max_author_degrees[author_node]:
        continue

    combined_tensor = torch.cat((co_author_candidates_history[author_node], co_author_candidates_top[author_node]), dim=0)
    coauthor_candidates = torch.unique(combined_tensor, dim=0).to(DEVICE)

    # Filtriamo solo i candidati con gradi disponibili
    valid_mask = (coauthor_candidates > author_node) & \
                 (current_author_degrees[coauthor_candidates] < max_author_degrees[coauthor_candidates])
    valid_candidates = coauthor_candidates[valid_mask]

    if valid_candidates.size(0) == 0:
        continue

    # Calcoliamo le probabilità per i candidati validi non ancora definite
    prob_mask = torch.tensor([probability_dict.get((author_node, candidate), 0) == 0 for candidate in valid_candidates])
    new_probs = connection_probability(author_node, valid_candidates[prob_mask]).clamp(min=0, max=1)
    
    # Aggiungiamo le nuove probabilità al dizionario
    for i, candidate in enumerate(valid_candidates[prob_mask]):
        probability_dict[(author_node, candidate.item())] = new_probs[i].item()

    # Estrazione delle connessioni da campionare in batch
    connections = torch.tensor([probability_dict[(author_node, candidate.item())] for candidate in valid_candidates])
    
    # Applica connessioni dove necessario
    connected_nodes = valid_candidates[connections > 0.5]
    for coauthor_node in connected_nodes:
        # Aggiorna l'edge index e i gradi correnti
        edge_index = torch.cat((edge_index, torch.tensor([[author_node], [coauthor_node]], device=DEVICE)), dim=1)
        current_author_degrees[author_node] += 1
        current_author_degrees[coauthor_node] += 1

        # Controlla se è stato raggiunto il grado massimo per l'autore corrente
        if current_author_degrees[author_node] >= max_author_degrees[author_node]:
            break

    # Stima del tempo rimanente ogni 10k autori
    if (j - 1) % 1000 == 0:
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (j + 1)) * (len(filtered_authors) - j - 1)
        print(f"Processed {j + 1}/{len(filtered_authors)} authors. Estimated time remaining: {remaining_time / 60:.2f} minutes")

# Salvataggio delle strutture su disco
path = f'../anp_data/processed/gt_edge_index_{infosphere_type}_{infosphere_parameters}_{YEAR}.pt'
torch.save(edge_index, path)
path = f'../anp_data/processed/gt_probability_dict_{infosphere_type}_{infosphere_parameters}_{YEAR}.pt'
torch.save(probability_dict, path)


print("Matrices saved successfully.")

