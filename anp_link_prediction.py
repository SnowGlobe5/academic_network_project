import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
from tqdm import tqdm

from torch_geometric.loader import HGTLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero
from anp_dataset import ANPDataset
from anp_dataloader import ANPDataLoader
from anp_utils import generate_coauthor_edge_year, anp_filter_data

BATCH_SIZE = 4096
YEAR_TRAIN = 2019
YEAR_VAL = 2020

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

root = "ANP_DATA"

dataset = ANPDataset(root=root)
data = dataset[0]
data['paper'].x = data['paper'].x.to(torch.float)

sub_graph_train, sub_graph_val, _, _ = anp_filter_data(data, root=root, fold=-1, max_year=YEAR_TRAIN, keep_edges=False)

## Train
sub_graph_train.to(device)
sub_graph_train = T.ToUndirected()(sub_graph_train)
train_input_nodes = ('author', torch.ones(sub_graph_train['author'].num_nodes, dtype=torch.bool))

## Validation
sub_graph_val.to(device)
sub_graph_val = T.ToUndirected()(sub_graph_val)
val_input_nodes = ('author', torch.ones(sub_graph_val['author'].num_nodes, dtype=torch.bool))


kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 20, 'persistent_workers': True}

train_loader = HGTLoader(sub_graph_train, num_samples=[4096] * 4, shuffle=True,
                            input_nodes=train_input_nodes, **kwargs)
val_loader = HGTLoader(sub_graph_val, num_samples=[4096] * 4, shuffle=True,
                            input_nodes=val_input_nodes, **kwargs)

data = next(iter(train_loader))
generate_coauthor_edge_year(data, YEAR_TRAIN)
del data['paper', 'rev_writes', 'author']
del data['topic', 'rev_about', 'paper']

weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


# The encoder (e.g. 2-layer GNN) should operate on the full graph (including all node and edge types we have). 
# The encoding is a dictionary of node_types and their embedding matrices. We will only use the embedding matrix for the node_type=author. 
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# We can then pick the embedding pairs for the node-pairs that are in our training set of pairs and feed them batch wise into the decoder (e.g. 2 layer fully connected NN with binary output for co-author/not-co-author).
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
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    generate_coauthor_edge_year(batch, YEAR_TRAIN)
    batch['author'].x = torch.eye(batch['author'].num_nodes, device=device)
    del batch['author'].num_nodes
    
    init_data, _, _ = T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        neg_sampling_ratio=1.0,
        edge_types=[('author', 'co_author', 'author')],
    )(batch)

    model(init_data.x_dict, init_data.edge_index_dict, init_data['author', 'author'].edge_label_index)


def train():
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        if i == 200: break
        generate_coauthor_edge_year(batch, YEAR_TRAIN)

        # Add user node features for message passing:
        batch['author'].x = torch.eye(batch['author'].num_nodes, device=device)
        del batch['author'].num_nodes
        del batch['paper', 'rev_writes', 'author']
        del batch['topic', 'rev_about', 'paper']
        batch.to(device)

        # Perform a link-level split into training, validation, and test edges:
        train_data, _, _ = T.RandomLinkSplit(
            num_val=0,
            num_test=0,
            neg_sampling_ratio=1.0,
            edge_types=[('author', 'co_author', 'author')],
        )(batch)
        
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                    train_data['author', 'author'].edge_label_index)
        target = train_data['author', 'author'].edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        total_examples += len(pred)
        total_loss += float(loss) * len(pred)

    return total_loss / total_examples
     
     
@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_mse = total_correct = 0
    for i, batch in enumerate(tqdm(loader)):
        if i == 200: break
        generate_coauthor_edge_year(batch, YEAR_TRAIN)

        # Add user node features for message passing:
        batch['author'].x = torch.eye(batch['author'].num_nodes, device=device)
        del batch['author'].num_nodes

        batch.to(device)
        del batch['paper', 'rev_writes', 'author']
        del batch['topic', 'rev_about', 'paper']
        
        # Perform a link-level split into training, validation, and test edges:
        val_data, _, _ = T.RandomLinkSplit(
            num_val=0,
            num_test=0,
            neg_sampling_ratio=1.0,
            edge_types=[('author', 'co_author', 'author')],
        )(batch)
        
        pred = model(val_data.x_dict, val_data.edge_index_dict,
                 val_data['author', 'author'].edge_label_index)
        pred = pred.clamp(min=0, max=1)
        target = val_data['author', 'author'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        total_mse += rmse
        total_examples += len(pred)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

    return total_mse / BATCH_SIZE, total_correct/total_examples


#init_params()  # Initialize parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 11):
    loss = train()
    val_mse, val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, RMSE: {val_mse:.4f}, Accuracy: {val_acc:.4f}')