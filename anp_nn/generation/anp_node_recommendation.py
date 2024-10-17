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
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import coalesce
from tqdm import tqdm

# Constants
BATCH_SIZE = 4096
YEAR = 2019
ROOT = "../anp_data"
DEVICE = torch.device(f'cuda:{sys.argv[7]}' if torch.cuda.is_available() else 'cpu')
from academic_network_project.anp_core.anp_dataset import ANPDataset
from academic_network_project.anp_core.anp_utils import *

# Get command line arguments
learning_rate = float(sys.argv[1])
infosphere_type = int(sys.argv[2])
infosphere_parameters = sys.argv[3]
only_new = sys.argv[4].lower() == 'true'
edge_number = int(sys.argv[5])
drop_percentage = float(sys.argv[6])

# Current timestamp for model saving
current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PATH = f"../anp_models/{os.path.basename(sys.argv[0][:-3])}_{infosphere_type}_{infosphere_parameters}_{only_new}_{edge_number}_{drop_percentage}_{current_date}/"
os.makedirs(PATH)
with open(PATH + 'info.json', 'w') as json_file:
    json.dump({'lr': learning_rate, 'infosphere_type': infosphere_type, 'infosphere_parameters': infosphere_parameters,
               'only_new': only_new, 'edge_number': edge_number, 'drop_percentage': drop_percentage, 'data': []}, json_file)


# Create ANP dataset
dataset = ANPDataset(root=ROOT)
data = dataset[0]

# Add infosphere data if requested
if infosphere_type != 0:
    if infosphere_type == 1:
        fold = [0, 1, 2, 3, 4]
        fold_string = '_'.join(map(str, fold))
        name_infosphere = f"{infosphere_parameters}_infosphere_{fold_string}_{YEAR}_noisy.pt"

        # Load infosphere
        if os.path.exists(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}"):
            infosphere_edges = torch.load(f"{ROOT}/computed_infosphere/{YEAR}/{name_infosphere}", map_location=DEVICE)
            
             # Drop edges for each type of relationship
            cites_edges = drop_edges(infosphere_edges[CITES], drop_percentage)
            writes_edges = drop_edges(infosphere_edges[WRITES], drop_percentage)
            about_edges = drop_edges(infosphere_edges[ABOUT], drop_percentage)
    
            data['paper', 'infosphere_cites', 'paper'].edge_index = coalesce(cites_edges)
            data['paper', 'infosphere_cites', 'paper'].edge_label = None
            data['author', 'infosphere_writes', 'paper'].edge_index = coalesce(writes_edges)
            data['author', 'infosphere_writes', 'paper'].edge_label = None
            data['paper', 'infosphere_about', 'topic'].edge_index = coalesce(about_edges)
            data['paper', 'infosphere_about', 'topic'].edge_label = None
        else:
            raise Exception(f"{name_infosphere} not found!")
        
    elif infosphere_type == 2:
        infosphere_edge = create_infosphere_top_papers_edge_index(data, int(infosphere_parameters), YEAR)
        data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere', 'paper'].edge_label = None

    elif infosphere_type == 3:
        infosphere_parameterss = infosphere_parameters.strip()
        arg_list = ast.literal_eval(infosphere_parameterss)
        if os.path.exists(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt"):
            print("Infosphere 3 edge found!")
            data['author', 'infosphere', 'paper'].edge_index = torch.load(f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt", map_location=DEVICE)
            data['author', 'infosphere', 'paper'].edge_label = None
        else:
            print("Generating infosphere 3 edge...")
            infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
            data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
            data['author', 'infosphere', 'paper'].edge_label = None
            torch.save(data['author', 'infosphere', 'paper'].edge_index, f"{ROOT}/processed/edge_infosphere_3_{arg_list[0]}_{arg_list[1]}.pt")

       
        infosphere_edge = create_infosphere_top_papers_per_topic_edge_index(data, arg_list[0], arg_list[1], YEAR)
        data['author', 'infosphere', 'paper'].edge_index = coalesce(infosphere_edge)
        data['author', 'infosphere', 'paper'].edge_label = None

# Try to predict all the future co-author or just the new one (not present in history)
label_function = get_difference_author_edge_year if only_new else get_author_edge_year
label_year = YEAR if only_new else YEAR + 1
label_file = f"{ROOT}/processed/difference_author_edge{label_year}.pt" if only_new \
    else f"{ROOT}/processed/author_edge{label_year}.pt"

# Use existing co-author edge if available, else generate
if os.path.exists(label_file):
    print("Label edge found!")
    labels = torch.load(label_file, map_location=DEVICE)
else:
    print("Label edge...")
    author_edge = label_function(data, label_year, DEVICE)
    labels = author_edge
    torch.save(author_edge, label_file)

# Convert paper features to float and make the graph undirected
data['paper'].x = data['paper'].x.to(torch.float)
data = T.ToUndirected()(data)
data = data.to('cpu')

sub_graph_train = anp_simple_filter_data(data, root=ROOT, folds=[0, 1, 2, 3], max_year=YEAR)
train_loader = NeighborLoader(
    data=sub_graph_train,
    num_neighbors=[edge_number, 30],
    batch_size=BATCH_SIZE,
    input_nodes=('author')
)

sub_graph_val = anp_simple_filter_data(data, root=ROOT, folds=[4], max_year=YEAR)
val_loader = NeighborLoader(
    data=sub_graph_val,
    num_neighbors=[edge_number, 30],
    batch_size=BATCH_SIZE,
    input_nodes=('author')
)

# Define model components
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                            num_heads)
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class NodeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, input_nodes):
        z_input = z_dict['author'][input_nodes]
        z_all = z_dict['author']
        scores = torch.matmul(self.lin(z_input), z_all.t())
        return F.sigmoid(scores)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, 2, 1, data.metadata())
        self.decoder = NodeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, input_nodes):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, input_nodes)

# Initialize model, optimizer, and embeddings
model = Model(hidden_channels=32).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
embedding_author = torch.nn.Embedding(data["author"].num_nodes, 32).to(DEVICE)
embedding_topic = torch.nn.Embedding(data["topic"].num_nodes, 32).to(DEVICE)


def train():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(DEVICE)
        original_ids = batch['author'].id
        optimizer.zero_grad()
        input_nodes = torch.arange(0, BATCH_SIZE, device=DEVICE)
        
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)         
        
        pred = model(batch.x_dict, batch.edge_index_dict, input_nodes)
        
        target = torch.zeros_like(pred)
        for node in input_nodes:
            mask = (labels["author"][0] == original_ids[node])
            future_coauthors_ori_id = labels["author"][1][mask]
            future_coauthors = torch.nonzero(original_ids.unsqueeze(1) == future_coauthors_ori_id.unsqueeze(0), as_tuple=False)[:, 0]
            target[node, future_coauthors] = 1
        
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss)
    
    return total_loss / (len(train_loader) * BATCH_SIZE)

@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    for batch in tqdm(loader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        co_authors = batch['author', 'author'].edge_index
        del batch['author', 'author']
        
        input_nodes = torch.arange(0, BATCH_SIZE, device=DEVICE)
        
        batch['author'].x = embedding_author(batch['author'].n_id)
        batch['topic'].x = embedding_topic(batch['topic'].n_id)         
        
        pred = model(batch.x_dict, batch.edge_index_dict, input_nodes)
        
        target = torch.zeros_like(pred)
        for i, node in enumerate(input_nodes):
            mask = (co_authors[0] == node)
            future_coauthors = co_authors[1][mask]
            target[i, future_coauthors] = 1
        
        loss = F.binary_cross_entropy(pred, target)
        total_loss += float(loss)
    
    return total_loss / (len(loader) * BATCH_SIZE)


# Main Training Loop
training_loss_list = []
validation_loss_list = []
best_val_loss = np.inf
patience = 5
counter = 0

# Training Loop
for epoch in range(1, 100):
    train_loss = train()
    val_loss = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}')

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # anp_save(model, PATH, epoch, train_loss, val_loss, val_acc)
        counter = 0  # Reset the counter if validation loss improves
    else:
        counter += 1
        if counter >= 5: 
            lr_scheduler.step(val_loss)

    # Early stopping check
    if counter >= patience and epoch >= 20:
        print(f'Early stopping at epoch {epoch}.')
        break

    training_loss_list.append(train_loss)
    validation_loss_list.append(val_loss)
    
# ... (il resto del codice rimane simile, con adattamenti per il nuovo modello)

# Funzione per predire le connessioni future più probabili
# def predict_future_connections(node_id, top_k=10):
#     model.eval()
#     with torch.no_grad():
#         batch = next(iter(NeighborLoader(data, num_neighbors=[-1], batch_size=1, input_nodes=('author', node_id))))
#         batch = batch.to(DEVICE)
#         pred = model(batch.x_dict, batch.edge_index_dict, 0)  # 0 perché c'è solo un nodo di input
        
#         # Ottieni i top_k nodi più probabili
#         top_k_values, top_k_indices = torch.topk(pred, k=top_k)
        
#         return list(zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()))

# # Esempio di utilizzo
# node_id = 123  # ID del nodo di input
# future_connections = predict_future_connections(node_id)
# print(f"Le {len(future_connections)} connessioni future più probabili per il nodo {node_id} sono:")
# for index, probability in future_connections:
#     print(f"Nodo {index}: probabilità {probability:.4f}")