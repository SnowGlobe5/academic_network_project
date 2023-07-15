import torch
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from anp_dataset import ANPDataset
from anp_utils import *
from torch.nn import Linear
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm

BATCH_SIZE = 4096
YEAR_TRAIN = 2019
YEAR_VAL = 2020

# Check if CUDA is available, else use CPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

ROOT = "ANP_DATA"
PATH = "ANP_MODELS/1_co_author_prediction/"

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

if os.path.exists(f"{ROOT}/processed/co_author_edge{YEAR_TRAIN}.pt"):
    print("Co-author edge found!")
    data['author', 'co_author', 'author'].edge_index = torch.load(f"{ROOT}/processed/co_author_edge{YEAR_TRAIN}.pt")
    data['author', 'co_author', 'author'].edge_label = None
else:
    print("Generating co-author edge...")
    generate_coauthor_edge_year(data, YEAR_TRAIN)
    torch.save(data['author', 'co_author', 'author'].edge_index, f"{ROOT}/processed/co_author_edge{YEAR_TRAIN}.pt")

data['paper'].x = data['paper'].x.to(torch.float)

# Train
# Filter training data
sub_graph_train, _, _, _ = anp_filter_data(data, root=ROOT, folds=[0, 1, 2, 3 ], max_year=YEAR_TRAIN, keep_edges=False)    
sub_graph_train = sub_graph_train.to(device)
sub_graph_train = T.ToUndirected()(sub_graph_train)

# Validation
# Filter validation data
sub_graph_val, _, _, _ = anp_filter_data(data, root=ROOT, folds=[4], max_year=YEAR_TRAIN, keep_edges=False)
sub_graph_val = sub_graph_val.to(device)
sub_graph_val = T.ToUndirected()(sub_graph_val)

# Set loader parameters
kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 6, 'persistent_workers': True}

# Create train and validation loaders
train_loader = HGTLoader(sub_graph_train, num_samples=[4096] * 4, shuffle=True,
                            input_nodes='author', **kwargs)
val_loader = HGTLoader(sub_graph_val, num_samples=[4096] * 4, shuffle=True,
                            input_nodes='author', **kwargs)


# Initialize weight
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


# Create model, optimizer, and move model to device
model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
embedding = torch.nn.Embedding(data["author"].num_nodes, 32)

if os.path.exists(PATH):
    model, first_epoch = anp_load(model, PATH) + 1
else:
    os.makedirs(PATH)
    with open(PATH + 'info.json', 'w') as json_file:
        json.dump([], json_file)
    first_epoch = 1
    

def train():
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        # Add user node features for message passing:
        # batch['author'].x = torch.eye(batch['author'].num_nodes, device=device)
        batch['author'].x = embedding(batch['author'].n_id)

        del batch['paper', 'rev_writes', 'author']
        del batch['topic', 'rev_about', 'paper']
        batch = batch.to(device)

        # Add 0/1 features to co_author edge:
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
        # Add user node features for message passing:
        batch['author'].x = embedding(batch['author'].n_id)

        batch = batch.to(device)
        del batch['paper', 'rev_writes', 'author']
        del batch['topic', 'rev_about', 'paper']

        # # Add 0/1 features to co_author edge:
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

    return total_mse / BATCH_SIZE, total_correct / total_examples


# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(first_epoch, 31):
    # Train the model
    loss = train()

    # Test the model
    val_mse, val_acc = test(val_loader)

    anp_save(model, PATH, epoch, loss, val_mse.item(), val_acc)
    
    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, RMSE: {val_mse:.4f}, Accuracy: {val_acc:.4f}')
    