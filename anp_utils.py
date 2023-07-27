import torch
import json
from datetime import datetime
import pandas as pd

PAPER = 0
AUTHOR = 1
TOPIC = 2

CITES = 0
WRITES = 1
ABOUT = 2

MAX_ITERATION = 1
DEVICE = 'cuda:1'


def expand_1_hop_edge_index(edge_index, node, flow):
    # _, sub_edge_index, _, _ = k_hop_subgraph(node, 1, edge_index, flow=flow)
    # Clean
    if flow == 'target_to_source':
        mask = edge_index[0] == node
    else:
        mask = edge_index[1] == node
    return edge_index[:, mask], mask


def expand_1_hop_graph(edge_index, nodes, type, paths):
    if type == PAPER:
        paper_nodes = []
        author_nodes = []
        topic_nodes = []
        for paper in nodes:
            # cited paper
            # sub_edge_index, _ = expand_1_hop_edge_index(edge_index[0], paper, flow='target_to_source')
            # for cited_paper in sub_edge_index[1].tolist():
            #     if not paths[PAPER].get(cited_paper):
            #         paper_nodes.append(cited_paper)
            #         paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [paper, cited_paper])]
            
            # Since this is used from seed we are intrested in papers that cites it
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[CITES], paper, flow='source_to_target')
            for cited_paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(cited_paper):
                    paper_nodes.append(cited_paper)
                    paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [cited_paper, paper])]

            # co-authors
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[WRITES], paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                if not paths[AUTHOR].get(co_author):
                    author_nodes.append(co_author)
                    paths[AUTHOR][co_author] = paths[PAPER][paper] + [('writes', [co_author, paper])]

            # topic
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[ABOUT], paper, flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if not paths[TOPIC].get(topic):
                    topic_nodes.append(topic)
                    paths[TOPIC][topic] = paths[PAPER][paper] + [('about', [paper, topic])]

        return (paper_nodes, author_nodes, topic_nodes)

    elif type == AUTHOR:
        paper_nodes = []
        for author in nodes:
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[AUTHOR][author] + [('writes', [author, paper])]
        return paper_nodes

    else:
        paper_nodes = []
        for topic in nodes:
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index, topic, flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[TOPIC][topic] + [('about', [paper, topic])]
        return paper_nodes
    
    
def anp_filter_data(data, root, folds, max_year, keep_edges):
        subset_dict = {}
        subset_dict_next_year = {}
        
        authors_filter_list = []
        for fold in folds:
            df_auth = pd.read_csv(f"{root}/split/authors_{fold}.csv")
            authors_filter_list.extend(df_auth.values.flatten())
        
        if not keep_edges:
            subset_dict['author'] = subset_dict_next_year['author'] = torch.tensor(authors_filter_list)    

        papers_list_next_year = []
        papers_list_year = []
        for i, row in enumerate(data['paper'].x.tolist()):
            if row[0] <= max_year:
                papers_list_year.append(i)
            elif row[0] == max_year + 1:
                papers_list_next_year.append(i)
        subset_dict['paper'] = torch.tensor(papers_list_year)
        papers_list_year.extend(papers_list_next_year)
        subset_dict_next_year['paper'] = torch.tensor(papers_list_year)
        return data.subgraph(subset_dict), data.subgraph(subset_dict_next_year), sorted(authors_filter_list), papers_list_next_year
    

def generate_co_author_edge_year(data, year):
    years = data['paper'].x[:, 0]
    mask = years == year
    papers = torch.where(mask)[0]
    edge_index = data['author', 'writes', 'paper'].edge_index
    edge_index = edge_index.to(DEVICE)
    src = []
    dst = []
    dict_tracker = {}
    time = datetime.now()
    tot = len(papers)
    for i, paper in enumerate(papers):
        if i % 10000 == 0:
            print(f"papers processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
        sub_edge_index, _ = expand_1_hop_edge_index(edge_index, paper, flow='source_to_target')
        for author in sub_edge_index[0].tolist():
            for co_author in sub_edge_index[0].tolist():
                if author != co_author and not dict_tracker.get((author, co_author)):
                    dict_tracker[(author, co_author)] = True
                    src.append(author)
                    dst.append(co_author)
    data['author', 'co_author', 'author'].edge_index = torch.tensor([src, dst])
    data['author', 'co_author', 'author'].edge_label = None
    
    
def generate_next_topic_edge_year(data, year):
    years = data['paper'].x[:, 0]
    mask = years == year
    papers = torch.where(mask)[0]
    edge_index_writes = data['author', 'writes', 'paper'].edge_index
    edge_index_writes = edge_index_writes.to(DEVICE)
    edge_index_about = data['paper', 'about', 'topic'].edge_index
    edge_index_about = edge_index_about.to(DEVICE)
    src = []
    dst = []
    dict_tracker = {}
    time = datetime.now()
    tot = len(papers)
    for i, paper in enumerate(papers):
        if i % 10000 == 0:
            print(f"papers processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
        sub_edge_index_writes, _ = expand_1_hop_edge_index(edge_index_writes, paper, flow='source_to_target')
        sub_edge_index_about, _ = expand_1_hop_edge_index(edge_index_about, paper, flow='target_to_source')
        for author in sub_edge_index_writes[0].tolist():
            for topic in sub_edge_index_about[1].tolist():
                if not dict_tracker.get((author, topic)):
                    dict_tracker[(author, topic)] = True
                    src.append(author)
                    dst.append(topic)
    data['author', 'next_topic', 'topic'].edge_index = torch.tensor([src, dst])
    data['author', 'next_topic', 'topic'].edge_label = None
    
    
def anp_save(model, path, epoch, loss, mse, accuracy):
    torch.save(model, path + 'model.pt')
    new = {
        'epoch': epoch,
        'loss': loss,
        'mse': mse,
        'accuracy': accuracy
    }
    with open(path + 'info.json', 'r') as json_file:
        data = json.load(json_file)
    data.append(new)
    with open(path + 'info.json', 'w') as json_file:
        json.dump(data, json_file)
        
def anp_load(path):
    with open(path + 'info.json', 'r') as json_file:
        data = json.load(json_file)
    return torch.load(path + 'model.pt'), data[-1]["epoch"]
    