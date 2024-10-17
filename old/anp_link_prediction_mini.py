import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from academic_network_project.anp_core.anp_dataset import ANPDataset
from anp_utils import get_author_edge_year, anp_filter_data

current_date = datetime.now().strftime("%Y-%m-%d")

BATCH_SIZE = 4096
YEAR_TRAIN = 2019
YEAR_VAL = 2020

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

ROOT = "../anp_data"

dataset = ANPDataset(root=root)
data = dataset[0]
data['paper'].x = data['paper'].x.to(torch.float)

## Train
sub_graph_mini, _, _, _ = anp_filter_data(data, root=root, folds=[0, 1, 2, 3, 4], max_year=YEAR_TRAIN, keep_edges=False)
sub_graph_mini.to(device)
mini_input_nodes = ('author', torch.ones(sub_graph_mini['author'].num_nodes, dtype=torch.bool))
sub_graph_mini = T.ToUndirected()(sub_graph_mini)

# kwargs = {'batch_size': 128, 'num_workers': 6, 'persistent_workers': True}

mini_loader = HGTLoader(sub_graph_mini, num_samples=[4096] * 4, shuffle=True,
                            input_nodes=mini_input_nodes, batch_size=BATCH_SIZE)

data = next(iter(mini_loader))
get_author_edge_year(data, YEAR_TRAIN)
data['author'].x = torch.eye(data['author'].num_nodes, device=device)
del data['author'].num_nodes

del data['paper', 'rev_writes', 'author']
del data['topic', 'rev_about', 'paper']

data.to(device)

# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,
    edge_types=[('author', 'co_author', 'author')],
)(data)

weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


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


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['author', 'author'].edge_label_index)
    target = train_data['author', 'author'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['author', 'author'].edge_label_index)
    pred = pred.clamp(min=0, max=1)
    target = data['author', 'author'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


for epoch in range(1, 151):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')