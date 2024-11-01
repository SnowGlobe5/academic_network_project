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
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

import torch
import pandas as pd

# Definisci il range temporale per filtrare i paper (es. 2015-2020)
YEAR_1, YEAR_2 = 2015, 2020
mask = (data['paper'].x[:, 0] >= YEAR_1) & (data['paper'].x[:, 0] <= YEAR_2)

# Applica il mask per ottenere gli ID dei paper nel range specificato
papers_list_year = set(torch.where(mask)[0].tolist())
print(f"Number of papers between {YEAR_1} and {YEAR_2}: {len(papers_list_year)}")

# Carica i paper popolari fino al 2019
df = pd.read_csv("../anp_data/raw/sorted_papers.csv")
popular_papers = set(df[df['year'] <= 2019]['id'][:50].values.tolist())
print(f"Number of popular papers: {len(popular_papers)}")

# Carica i topic e seleziona i 50 più popolari
df_topic = pd.read_csv(f"../anp_data/raw/sorted_authors_topics_{2019}.csv")
top_topics = df_topic['topic_id'].value_counts().head(50).index.tolist()

# Filtra i paper per anno e topic popolari, poi seleziona i 5 più citati per topic
df_papers = pd.read_csv("../anp_data/raw/sorted_papers_about.csv")
df_papers_filtered = df_papers[(df_papers['year'] <= 2019) & (df_papers['topic_id'].isin(top_topics))]
df_papers_filtered = df_papers_filtered.sort_values(by='citations', ascending=False)
top_cited_papers = set(df_papers_filtered.groupby('topic_id').head(5)['id'].tolist())
print(f"Number of papers from popular topics: {len(top_cited_papers)}")

# Unisci tutti i paper in un set unico
all_relevant_papers = papers_list_year | popular_papers | top_cited_papers
print(f"Total unique papers: {len(all_relevant_papers)}")

# Filtra autori dei paper
author_paper = data['author', 'writes', 'paper'].edge_index

# Prendiamo gli autori collegati ai paper nel set 'all_relevant_papers'
all_relevant_papers_tensor = torch.tensor(list(all_relevant_papers))
paper_mask = torch.isin(author_paper[1], all_relevant_papers_tensor)
relevant_edges = author_paper[:, paper_mask]

# Prendiamo gli autori unici collegati ai paper rilevanti
authors_set = set(relevant_edges[0].tolist())
print(f"Number of unique authors for relevant papers: {len(authors_set)}")

# Converti il set di autori in un DataFrame
authors_df = pd.DataFrame(list(authors_set), columns=['author_id'])

# Salva il DataFrame su disco come file CSV
authors_df.to_csv("../anp_data/processed/relevant_authors.csv", index=False)
print("Author IDs saved to '../anp_data/processed/relevant_authors.csv'")
