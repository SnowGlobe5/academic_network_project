import torch
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
from anp_dataset import ANPDataset
from anp_utils import *
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm

BATCH_SIZE = 4096
YEAR = 2019

ROOT = "ANP_DATA"

DEVICE=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

fold = [0, 1, 2, 3, 4] #TODO param
fold_string = [str(x) for x in fold]
fold_string = '_'.join(fold_string)
name_infosphere = f"5_infosphere_{fold_string}_{YEAR}_noisy.pt"

# Get infosphere
if os.path.exists(f"{ROOT}/computed_infosphere/{name_infosphere}"):
    infosphere_edges = torch.load(f"{ROOT}/computed_infosphere/{name_infosphere}")
    data['paper', 'infosphere_cites', 'paper'].edge_index = coalesce(infosphere_edges[CITES])
    data['paper', 'infosphere_cites', 'paper'].edge_label = None
    data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(infosphere_edges[WRITES])
    data['author', 'infosphere_writes', 'paper'].edge_label = None
    data['paper', 'infosphere_about', 'topic'].edge_index = coalesce(infosphere_edges[ABOUT])
    data['paper', 'infosphere_about', 'topic'].edge_label = None
else:
    raise Exception(f"{name_infosphere} not found!!")


# Use already existing co-author edge (if exist)
if os.path.exists(f"{ROOT}/processed/co_author_edge{YEAR+1}.pt"):
    print("Co-author edge found!")
    data['author', 'co_author', 'author'].edge_index = torch.load(f"{ROOT}/processed/co_author_edge{YEAR+1}.pt")
    data['author', 'co_author', 'author'].edge_label = None
else:
    print("Generatinge co-author edge...")
    data['author', 'co_author', 'author'].edge_index = generate_co_author_edge_year(data, YEAR+1, ROOT)
    data['author', 'co_author', 'author'].edge_label = None
    torch.save(data['author', 'co_author', 'author'].edge_index, f"{ROOT}/processed/co_author_edge{YEAR+1}.pt")

# Make paper features float and the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

# Validation
# Filter validation data
sub_graph_val= anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)  
#sub_graph_val = sub_graph_val.to(DEVICE)

transform = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    #neg_sampling_ratio=2.0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=('author', 'co_author', 'author')
)
val_data, _, _= transform(sub_graph_val)


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

# Delete the co-author edge (data will be used for data.metadata())
del data['author', 'co_author', 'author']

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

model = torch.load("ANP_MODELS/1_co_author_prediction_future_no_info/model.pt")
embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)

with open('history4co2.json', 'r') as json_file:
        author_json = json.load(json_file)

@torch.no_grad()
def test(loader):
    model.eval()
    for i, batch in enumerate(tqdm(loader)):
        batch = batch.to(DEVICE)
        
        edge_label_index = batch['author', 'co_author', 'author'].edge_label_index
        #print(edge_label_index)
        edge_label = batch['author', 'co_author', 'author'].edge_label
        del batch['author', 'co_author', 'author']
        
        # Add user node features for message passing:
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)

        pred = model(batch.x_dict, batch.edge_index_dict, edge_label_index)
        target = edge_label
        pred = pred.clamp(min=0, max=1)

        
        for i in range(len(target)):
            author_1_id = int(batch['author'].id[int(edge_label_index[0][i].item())].item())
            author_2_id = int(batch['author'].id[int(edge_label_index[1][i].item())].item())
            #print(author_json[f"{author_1_id}"]["co-authors"])
            # if not author_json[f"{author_1_id}"]["co-authors"] or not author_json[f"{author_2_id}"]["co-authors"]:
            #      continue
            if target[i].item() == 1:
                if torch.round(pred, decimals=0)[i].item() == 1:
                    if author_2_id in author_json[f"{author_1_id}"]["co-authors"]:
                        confusion_matrix['correct_history'] += 1
                        #print(author_1_id, author_2_id)
                        #print(author_json[f"{author_1_id}"])
                    else:
                        confusion_matrix['correct_new'] += 1
                else:
                    if author_2_id in author_json[f"{author_1_id}"]["co-authors"]:
                        confusion_matrix['incorrect_history'] += 1
                    else:
                        confusion_matrix['incorrect_new'] += 1



confusion_matrix = {
    'correct_history': 0,
    'incorrect_history': 0,
    'correct_new': 0,
    'incorrect_new': 0,
    'no_history': 0
}

test(val_loader)

print(confusion_matrix)