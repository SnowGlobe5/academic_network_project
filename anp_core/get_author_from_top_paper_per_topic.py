import json
import os
import sys
import ast
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm
import time

# Constants
BATCH_SIZE = 4096
YEAR = 2019
NUM_PAPER= 2
NUM_TOPIC = 5
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]
data = data.to(DEVICE)

import torch
import pandas as pd

path = f'../anp_data/processed/co_author_edge{YEAR}_history.pt'
co_author_edges_history = torch.load(path).to(DEVICE)

# Carica i paper popolari fino al YEAR
author_top_paper_topic_path = f"../anp_data/processed/edge_infosphere_3_{NUM_TOPIC}_{NUM_PAPER}.pt"
author_top_paper_topic = torch.load(author_top_paper_topic_path).to(DEVICE)

# Get mask of connections in `writes` involving popular papers
writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)

list_co_author_edges = []
start_time = time.time()  # Registra l'ora di inizio

for author in range(data["author"].num_nodes):
    if author % 10000 == 0:
        elapsed_time = time.time() - start_time  # Tempo trascorso in secondi
        percent_complete = (author / data["author"].num_nodes) * 100
        
        # Calcola il tempo stimato rimanente
        if author > 0:
            time_per_author = elapsed_time / author
            remaining_authors = data["author"].num_nodes - author
            estimated_time_remaining = remaining_authors * time_per_author
            estimated_minutes = estimated_time_remaining / 60
            
            print(f"Progress: {percent_complete:.2f}% completed")
            print(f"Estimated time remaining: {estimated_minutes:.2f} minutes")
        else:
            print(f"Progress: {percent_complete:.2f}% completed")
        
    popular_papers = author_top_paper_topic[0][author]
    
    writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)
    mask_popular_papers = torch.isin(writes_edges[0], popular_papers)
    author_popular_papers = writes_edges[1, mask_popular_papers]

    # Get authors connected to the authors of popular papers in `co_author_edges_history`
    mask_co_authors = torch.isin(co_author_edges_history[0], author_popular_papers)
    co_author_authors_popular_papers = co_author_edges_history[:, mask_co_authors]

    # Converti gli autori in insiemi e rimuovi duplicati
    author_popular_papers_set = set(author_popular_papers.cpu().numpy().flatten())
    co_author_authors_popular_papers_set = set(co_author_authors_popular_papers.cpu().numpy().flatten())

    # Unisci i due insiemi e rimuovi duplicati
    final_authors_set = author_popular_papers_set | co_author_authors_popular_papers_set

    final_authors_list = list(map(int, final_authors_set))
    
    list_co_author_edges.append(final_authors_list)

# Salva l'insieme finale su disco come 'co_author_top_paper_NUM.json'
output_path = f"../anp_data/processed/co_author_top_paper_per_topic_{NUM_TOPIC}_{NUM_PAPER}_{YEAR}.json"
with open(output_path, "w") as f:
    json.dump(list_co_author_edges, f)

print(f"Combined set saved to {output_path}")
