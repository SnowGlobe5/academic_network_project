import torch
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from anp_dataset import ANPDataset
from anp_utils import *
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm

BATCH_SIZE = 4096
YEAR = 2019

ROOT = "ANP_DATA"
PATH = "ANP_MODELS/1_co_author_prediction/"

#TODO remove
import shutil
try:
    shutil.rmtree(PATH)
except:
    pass

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Use already existing co-author edge (if exist)
if os.path.exists(f"{ROOT}/processed/co_author_edge{YEAR}.pt"):
    print("Co-author edge found!")
    data['author', 'co_author', 'author'].edge_index = torch.load(f"{ROOT}/processed/co_author_edge{YEAR}.pt")
    data['author', 'co_author', 'author'].edge_label = None
else:
    print("Generating co-author edge...")
    data['author', 'co_author', 'author'].edge_index = generate_co_author_edge_year(data, YEAR)
    data['author', 'co_author', 'author'].edge_label = None
    torch.save(data['author', 'co_author', 'author'].edge_index, f"{ROOT}/processed/co_author_edge{YEAR}.pt")

# Make paper features float and the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=('author', 'co_author', 'author')
)
train_data, val_data, _= transform(data)

# Define seed edges:
edge_label_index = train_data['author', 'co_author', 'author'].edge_label_index
edge_label = train_data['author', 'co_author', 'author'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=256,
    shuffle=True,
)

edge_label_index = val_data['author', 'co_author', 'author'].edge_label_index
edge_label = val_data['author', 'co_author', 'author'].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 256,
    shuffle=False,
)

# Delete the co-author edge (data will be used for data.metadata())
del data['author', 'co_author', 'author']

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
        self.conv3 = SAGEConv((-1, -1), out_channels)
        self.conv4 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['author'][row], z_dict['author'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z).relu()
        z = self.lin4(z)
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
# If exist load last checkpoint
if os.path.exists(PATH):
    model, first_epoch = anp_load(PATH)
    first_epoch += 1
else:
    model = Model(hidden_channels=32).to(DEVICE)
    os.makedirs(PATH)
    with open(PATH + 'info.json', 'w') as json_file:
        json.dump([], json_file)
    first_epoch = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)

def train():
    model.train()
    total_examples = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(DEVICE)
        
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        total_examples += len(pred)
        total_loss += float(loss) * len(pred)

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    total_examples = total_mse = total_correct = total_loss = 0
    for i, batch in enumerate(tqdm(loader)):
        batch = batch.to(DEVICE)
        
        edge_label_index = batch['author', 'author'].edge_label_index
        edge_label = batch['author', 'author'].edge_label
        del batch['author', 'co_author', 'author']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        pred = pred.clamp(min=0, max=1)
        target = edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        total_mse += rmse
        total_loss += float(loss) * len(pred)
        total_examples += len(pred)
        total_correct += int((torch.round(pred, decimals=0) == target).sum())

    return total_mse / BATCH_SIZE, total_correct / total_examples, total_loss / total_examples


# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_loss_list = []
validation_loss_list = []
accuracy_list = []

for epoch in range(first_epoch, 101):
    # Train the model
    loss = train()

    # Test the model
    val_mse, val_acc, loss_val = test(val_loader)

    # Save the model
    anp_save(model, PATH, epoch, loss, val_mse.item(), val_acc)
    
    training_loss_list.append(loss)
    validation_loss_list.append(loss_val)
    accuracy_list.append(val_acc)
    
    # Print epoch results
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, RMSE: {val_mse:.4f}, Accuracy: {val_acc:.4f}')
    
from matplotlib import pyplot as plt
plt.plot(training_loss_list, label='train_loss')
plt.plot(validation_loss_list,label='val_loss')
plt.legend()
plt.savefig('output/nll_link_split_loss.pdf')
plt.close()

plt.plot(accuracy_list,label='accuracy')
plt.legend()
plt.savefig('output/nll_link_split_accuracy.pdf')