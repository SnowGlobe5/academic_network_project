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

# Constants
BATCH_SIZE = 4096
YEAR = 2019
NUM = 10
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
df = pd.read_csv("../anp_data/raw/sorted_papers.csv")
popular_papers = set(df[df['year'] <= YEAR]['id'][:NUM].values.tolist())
print(f"Number of popular papers: {len(popular_papers)}")

# Carica i topic e seleziona i NUM più popolari
df_topic = pd.read_csv(f"../anp_data/raw/sorted_authors_topics_{YEAR}.csv")
top_topics = df_topic['topic_id'].value_counts().head(NUM).index.tolist()

# Convert popular papers to a tensor for filtering
popular_papers_tensor = torch.tensor(list(popular_papers), device=DEVICE)

# Get mask of connections in `writes` involving popular papers
writes_edges = data['author', 'writes', 'paper'].edge_index.to(DEVICE)
mask_popular_papers = torch.isin(writes_edges[0], popular_papers_tensor)
author_popular_papers = writes_edges[1, mask_popular_papers]

# Get authors connected to the authors of popular papers in `co_author_edges_history`
mask_co_authors = torch.isin(co_author_edges_history[0], author_popular_papers)
co_author_authors_popular_papers = co_author_edges_history[:, mask_co_authors]

# Converti gli autori in insiemi e rimuovi duplicati
author_popular_papers_set = set(author_popular_papers.cpu().numpy().flatten())
co_author_authors_popular_papers_set = set(co_author_authors_popular_papers.cpu().numpy().flatten())

# Stampa la lunghezza di ciascun insieme
print(f"Number of unique authors of popular papers: {len(author_popular_papers_set)}")
print(f"Number of unique co-authors of authors of popular papers: {len(co_author_authors_popular_papers_set)}")

# Unisci i due insiemi e rimuovi duplicati
final_authors_set = author_popular_papers_set | co_author_authors_popular_papers_set
print(f"Total unique authors in combined set: {len(final_authors_set)}")

final_authors_list = list(map(int, final_authors_set))

# Salva l'insieme finale su disco come 'co_author_top_paper_NUM.json'
output_path = f"../anp_data/processed/co_author_top_paper_{NUM}_{YEAR}.json"
with open(output_path, "w") as f:
    json.dump(final_authors_list, f)

print(f"Combined set saved to {output_path}")
