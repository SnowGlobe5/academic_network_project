import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear

from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
# # We initialize conference node features with a single one-vector as feature:
# dataset = DBLP(path, transform=T.Constant(node_types='conference'))
# data = dataset[0]
# print(data)

BATCH_SIZE = 4096
YEAR = 2019

ROOT = "../anp_data"

DEVICE=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]
data = anp_simple_filter_data(data, ROOT, [0], 2019)

# Make paper features float and the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to(DEVICE)


def create_masks(data, train_ratio=0.8, val_ratio=0.05, test_ratio=0.15):
    num_authors = data.num_nodes
    num_train = int(num_authors * train_ratio)
    num_val = int(num_authors * val_ratio)
    num_test = num_authors - num_train - num_val
    
    # Genera gli indici casuali degli autori
    indices = torch.randperm(num_authors)
    
    # Crea le maschere
    train_mask = torch.zeros(num_authors, dtype=torch.bool)
    val_mask = torch.zeros(num_authors, dtype=torch.bool)
    test_mask = torch.zeros(num_authors, dtype=torch.bool)
    
    # Assegna gli indici agli insiemi di train, validation e test
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    
    return train_mask, val_mask, test_mask

# Usa la funzione per creare le maschere
train_mask, val_mask, test_mask = create_masks(data)

embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)
data['author'].x = embedding_author(data['author'].id.long())
data['topic'].x = embedding_topic(data['topic'].id.long())

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            print(node_type)
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])


model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    print (train_mask, out[train_mask], data['author'].y[train_mask])
    loss = F.cross_entropy(out[train_mask], data['author'].y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in (train_mask, val_mask, test_mask):
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')