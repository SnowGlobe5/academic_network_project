import torch
import numpy as np
import json
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import os

PAPER = 0
AUTHOR = 1
TOPIC = 2

CITES = 0
WRITES = 1
ABOUT = 2

MAX_ITERATION = 1
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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
    
def anp_simple_filter_data(data, root, folds, max_year):
        subset_dict = {}
        authors_filter_list = []
        for fold in folds:
            df_auth = pd.read_csv(f"{root}/split/authors_{fold}.csv")
            authors_filter_list.extend(df_auth.values.flatten())
        subset_dict['author'] = torch.tensor(authors_filter_list)   
        mask = data['paper'].x[:, 0] <= max_year
        papers_list_year = torch.where(mask)
        subset_dict['paper'] = papers_list_year[0]
        return data.subgraph(subset_dict)
    
    
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
    return torch.tensor([src, dst])

def generate_co_author_edge_year_history(data, year):
    years = data['paper'].x[:, 0]
    mask = years <= year
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
    return torch.tensor([src, dst])
   
def generate_difference_co_author_edge_year_single(data, year, root):
    difference_edge_index = torch.tensor([[],[]]).to(torch.int64).to(DEVICE)
    # Use already existing co-author edge (if exist)
    if os.path.exists(f"{root}/processed/co_author_edge{year}.pt"):
        print("Current co-author edge found!")
        current_edge_index = torch.load(f"{root}/processed/co_author_edge{year}.pt")
    else:
        print("Generating current co-author edge...")
        current_edge_index =  generate_co_author_edge_year(data, year)
        torch.save(current_edge_index, f"{root}/processed/co_author_edge{year}.pt")
        
    if os.path.exists(f"{root}/processed/co_author_edge{year+1}.pt"):
        print("Next co-author edge found!")
        next_edge_index = torch.load(f"{root}/processed/co_author_edge{year+1}.pt")
    else:
        print("Generating next co-author edge...")
        next_edge_index =  generate_co_author_edge_year(data, year+1)
        torch.save(next_edge_index, f"{root}/processed/co_author_edge{year+1}.pt")
    
    time = datetime.now()
    tot = len(next_edge_index[0])
    for i in range(tot):
        if i % 10000 == 0:
            print(f"author edge processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
        mask = current_edge_index[0] == next_edge_index[0][i]
        if not next_edge_index[1][i] in current_edge_index[:, mask][1]:
            difference_edge_index = torch.cat((difference_edge_index, torch.Tensor([[next_edge_index[0][i]],[next_edge_index[1][i]]]).to(torch.int64).to(DEVICE)), dim=1)
    
    return difference_edge_index
  
def generate_difference_co_author_edge_year(data, year, root):
    difference_edge_index = torch.tensor([[],[]]).to(torch.int64).to(DEVICE)
    # Use already existing co-author edge (if exist)
    if os.path.exists(f"{root}/processed/co_author_edge{year}_history.pt"):
        print("Current history co-author edge found!")
        current_edge_index = torch.load(f"{root}/processed/co_author_edge{year}_history.pt")
    else:
        print("Generating current history co-author edge...")
        current_edge_index =  generate_co_author_edge_year_history(data, year)
        torch.save(current_edge_index, f"{root}/processed/co_author_edge{year}_history.pt")
        
    if os.path.exists(f"{root}/processed/co_author_edge{year+1}.pt"):
        print("Next co-author edge found!")
        next_edge_index = torch.load(f"{root}/processed/co_author_edge{year+1}.pt")
    else:
        print("Generating next co-author edge...")
        next_edge_index =  generate_co_author_edge_year(data, year+1)
        torch.save(next_edge_index, f"{root}/processed/co_author_edge{year+1}.pt")
    
    set_src = torch.unique(next_edge_index[0], sorted=True)
    time = datetime.now()
    tot = len(set_src)
    for i, src in enumerate(set_src):
        if i % 1000 == 0:
            print(f"author edge processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
        
        mask = current_edge_index[0] == src
        dst_old = current_edge_index[:, mask][1]
        
        mask = next_edge_index[0] == src
        dst_new = next_edge_index[:, mask][1]
        
        diff=dst_new[(dst_new.view(1, -1) != dst_old.view(-1, 1)).all(dim=0)]
        
        for dst in diff:
            difference_edge_index = torch.cat((difference_edge_index, torch.Tensor([[src],[dst]]).to(torch.int64).to(DEVICE)), dim=1)
            print(dst_old)
            print(dst_new)
            print(diff)
    return difference_edge_index
    
    
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
    return torch.tensor([src, dst])
    
    
def generate_difference_next_topic_edge_year(data, year, root):
    difference_edge_index = torch.tensor([[],[]]).to(torch.int64).to(DEVICE)
    # Use already existing next-topic edge (if exist)
    if os.path.exists(f"{root}/processed/next_topic_edge{year}.pt"):
        print("Current next-topic edge found!")
        current_edge_index = torch.load(f"{root}/processed/next_topic_edge{year}.pt")
    else:
        print("Generating current next-topic edge...")
        current_edge_index =  generate_next_topic_edge_year(data, year)
        torch.save(current_edge_index, f"{root}/processed/next_topic_edge{year}.pt")
        
    if os.path.exists(f"{root}/processed/next_topic_edge{year+1}.pt"):
        print("Next next-topic edge found!")
        next_edge_index = torch.load(f"{root}/processed/next_topic_edge{year+1}.pt")
    else:
        print("Generating next next-topic edge...")
        next_edge_index =  generate_next_topic_edge_year(data, year+1)
        torch.save(next_edge_index, f"{root}/processed/next_topic_edge{year+1}.pt")
    
    time = datetime.now()
    tot = len(next_edge_index[0])
    for i in range(tot):
        if i % 10000 == 0:
            print(f"author edge processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
        mask = current_edge_index[0] == next_edge_index[0][i]
        if not next_edge_index[1][i] in current_edge_index[:, mask][1]:
            difference_edge_index = torch.cat((difference_edge_index, torch.Tensor([[next_edge_index[0][i]],[next_edge_index[1][i]]]).to(torch.int64).to(DEVICE)), dim=1)
    
    return difference_edge_index

   
def anp_save(model, path, epoch, loss, loss_val, accuracy):
    torch.save(model, path + 'model.pt')
    new = {
        'epoch': epoch,
        'loss': loss,
        'mse': loss_val,
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


def generate_graph (training_loss_list, validation_loss_list, training_accuracy_list, validation_accuracy_list, confusion_matrix):
    time = datetime.now()
    plt.plot(training_loss_list, label='train_loss')
    plt.plot(validation_loss_list,label='validation_loss')
    plt.legend()
    plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_loss.pdf')
    plt.close()

    plt.plot(training_accuracy_list,label='train_accuracy')
    plt.plot(validation_accuracy_list,label='validation_accuracy')
    plt.legend()
    plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_accuracy.pdf')
    plt.close()
    
    # time = datetime.now()
    # plt.plot(training_loss_list[1:], label='train_loss')
    # plt.plot(validation_loss_list[1:],label='val_loss')
    # plt.legend()
    # plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_loss2.pdf')
    # plt.close()

    # plt.plot(accuracy_list[1:],label='accuracy')
    # plt.legend()
    # plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_accuracy2.pdf')
    # plt.close()
    
    # time = datetime.now()
    # plt.plot(training_loss_list[2:], label='train_loss')
    # plt.plot(validation_loss_list[2:],label='val_loss')
    # plt.legend()
    # plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_loss3.pdf')
    # plt.close()

    # plt.plot(accuracy_list[2:],label='accuracy')
    # plt.legend()
    # plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_accuracy3.pdf')
    # plt.close()


    array = [[confusion_matrix['tp'], confusion_matrix['fp']],[confusion_matrix['fn'], confusion_matrix['tn']]]
    df_cm = pd.DataFrame(array, index = [i for i in ("POSITIVE", "NEGATIVE")],
                    columns = [i for i in ("POSITIVE", "NEGATIVE")])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'out/{sys.argv[0][:-3]}_{time.strftime("%Y%m%d%H%M%S")}_CM.pdf')
    plt.close()
    
    value_log = {
        'training_loss_list': training_loss_list, 
        'validation_loss_list': validation_loss_list, 
        'training_accuracy_list': training_accuracy_list, 
        'validation_accuracy_list': validation_accuracy_list
    }
    
    with open(f'out/log_{time.strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(value_log, f, ensure_ascii=False, indent=4)
    