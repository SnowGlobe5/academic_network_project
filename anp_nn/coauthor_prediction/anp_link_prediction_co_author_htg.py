import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import DataLoader

from anp_dataset import ANPDataset
from anp_utils import *
from tqdm import tqdm

# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
# # We initialize conference node features with a single one-vector as feature:
# dataset = DBLP(path, transform=T.Constant(node_types='conference'))
# data = dataset[0]
# print(data)

BATCH_SIZE = 4096
YEAR = 2019

ROOT = "ANP_DATA"

DEVICE=torch.device('cuda')

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]
y = torch.randint(0, 4, (data["author"].num_nodes,), dtype=torch.long)
data["author"].y = y

# Use already existing co-author edge (if exist)
if os.path.exists(f"{ROOT}/processed/co_author_edge{YEAR+1}.pt"):
    print("Co-author edge found!")
    data['author', 'co_author', 'author'].edge_index = torch.load(f"{ROOT}/processed/co_author_edge{YEAR+1}.pt")
    data['author', 'co_author', 'author'].edge_label = None
else:
    print("Generating co-author edge...")
    data['author', 'co_author', 'author'].edge_index = generate_co_author_edge_year(data, YEAR+1)
    data['author', 'co_author', 'author'].edge_label = None
    torch.save(data['author', 'co_author', 'author'].edge_index, f"{ROOT}/processed/co_author_edge{YEAR+1}.pt")


# Make paper features float and the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

sub_graph_train= anp_simple_filter_data(data, root=ROOT, folds=[0, 1, 2, 3, 4], max_year=YEAR)    
 
transform = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0.0,
    #disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
train_data, val_data, _= transform(sub_graph_train)

# Define seed edges:
edge_label_index = train_data['author', 'co_author', 'author'].edge_label_index
edge_label = train_data['author', 'co_author', 'author'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    #neg_sampling_ratio=2.0,
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=True,
)

edge_label_index = val_data['author', 'co_author', 'author'].edge_label_index
edge_label = val_data['author', 'co_author', 'author'].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(('author', 'co_author', 'author'), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=False,
)

del data['author', 'co_author', 'author']


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
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

# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)

def train():
    model.train()
    total_examples = total_correct = total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(DEVICE)
        del batch['author', 'co_author', 'author']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)
        
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out, batch["author"].y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * out.numel()
        total_examples += out.numel()
        out = out.clamp(min=0, max=3)
        # total_correct += int((torch.round(out, decimals=0) == batch['author'].y).sum())
    return total_loss


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

#     accs = []
#     for split in (train_mask, val_mask, test_mask):
#         mask = data['author'][split]
#         acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
#         accs.append(float(acc))
#     return accs


for epoch in range(1, 101):
    loss = train()
    print(loss)
    # train_acc, val_acc, test_acc = test()
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
    #       f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')