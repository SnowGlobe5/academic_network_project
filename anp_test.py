from datetime import datetime

import torch
import cProfile
from torch_geometric.utils import k_hop_subgraph

from anp_dataloader import ANPDataLoader
from anp_dataset import ANPDataset

from enum import Enum

PAPER = 0
AUTHOR = 1
TOPIC = 2



def expand_1_hop_edge_index(edge_index, node, flow):
    # _, sub_edge_index, _, _ = k_hop_subgraph(node, 1, edge_index, flow=flow)
    # Clean
    if flow == 'target_to_source':
        mask = edge_index[0] == node
    else:
        mask = edge_index[1] == node
    return edge_index[:, mask]


def expand_1_hop_graph(edge_index, nodes, type):
    if type == PAPER:
        paper_nodes = set()
        author_nodes = set()
        topic_nodes = set()
        for paper in nodes:
            # cited paper
            sub_edge_index = expand_1_hop_edge_index(edge_index[0], paper, flow='target_to_source')
            for cited_paper in sub_edge_index[1].tolist():
                paper_nodes.add(cited_paper)

            # co-authors
            sub_edge_index = expand_1_hop_edge_index(edge_index[1], paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                author_nodes.add(co_author)

            # topic
            sub_edge_index = expand_1_hop_edge_index(edge_index[2], paper, flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                topic_nodes.add(topic)

        return (paper_nodes, author_nodes, topic_nodes)

    elif type == AUTHOR:
        paper_nodes = set()
        for author in nodes:
            sub_edge_index = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                paper_nodes.add(paper)
        return paper_nodes

    else:
        paper_nodes = set()
        for topic in nodes:
            sub_edge_index = expand_1_hop_edge_index(edge_index, topic, flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                paper_nodes.add(paper)
        return paper_nodes


def get_papers_per_author_year(data, author, papers_year_list):
    edge_index = data['author', 'writes', 'paper'].edge_index
    sub_edge_index = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
    mask = torch.isin(sub_edge_index[1], papers_year_list)
    return sub_edge_index[:, mask][1]


def get_history_infosphere(data, data_next_year, author_id, papers_next_year):
    author_papers_next_year = get_papers_per_author_year(data_next_year, author_id, papers_next_year)
    writes_edge_index_next_year = data_next_year['author', 'writes', 'paper'].edge_index
    cites_edge_index_next_year = data_next_year['paper', 'cites', 'paper'].edge_index
    about_edge_index_next_year = data_next_year['paper', 'about', 'topic'].edge_index

    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index

    infosphere = [{}, {}, {}]

    paper_user = set()
    author_user = set()
    topic_user = set()

    paper_seed = set()
    author_seed = set()
    topic_seed = set()

    paper_user_scanned = set()
    author_user_scanned = set()
    topic_user_scanned = set()

    paper_infosphere = set()
    author_infosphere = set()
    topic_infosphere = set()

    sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author_id, flow='target_to_source')
    author_papers = sub_edge_index[1].tolist()
    for paper in author_papers:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        for cited_paper in sub_edge_index[1].tolist():
            paper_user.add(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            author_user.add(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            topic_user.add(topic)

    for paper in author_papers_next_year:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index_next_year, paper, flow='target_to_source')
        mask = torch.isin(sub_edge_index[1], papers_next_year, invert=True)
        for cited_paper in sub_edge_index[:, mask][1].tolist():
            paper_seed.add(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index_next_year, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            author_seed.add(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index_next_year, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            topic_seed.add(topic)

    
    paper_seed.difference_update(paper_user)
    author_seed.difference_update(author_user)
    topic_seed.difference_update(topic_user)

    for i in range(1):
        if not (paper_seed or author_seed or topic_seed):
            break
        
        paper_infosphere.update(paper_seed)
        author_infosphere.update(author_seed)
        topic_infosphere.update(topic_seed)
        (temp_paper1, temp_author, temp_topic) = expand_1_hop_graph((cites_edge_index, writes_edge_index, about_edge_index), paper_seed, PAPER)
        temp_paper2 = expand_1_hop_graph(writes_edge_index, author_seed, AUTHOR)
        # temp_paper3 = expand_1_hop_graph(about_edge_index, topic_seed, TOPIC)

        paper_seed.update(temp_paper1)
        paper_seed.update(temp_paper2)
        # paper_seed.update(temp_paper3)
        author_seed.update(temp_author)
        topic_seed.update(temp_topic)

        paper_seed.difference_update(paper_infosphere)
        author_seed.difference_update(author_infosphere)
        topic_seed.difference_update(topic_infosphere)

        paper_seed.difference_update(paper_user)
        author_seed.difference_update(author_user)
        topic_seed.difference_update(topic_user)
    
        if not (paper_seed or author_seed or topic_seed):
            break
    
        paper_user_scanned.update(paper_user)
        author_user_scanned.update(author_user)
        topic_user_scanned.update(topic_user)
        (temp_paper1, temp_author, temp_topic) = expand_1_hop_graph((cites_edge_index, writes_edge_index, about_edge_index), paper_user, PAPER)
        temp_paper2 = expand_1_hop_graph(writes_edge_index, author_user, AUTHOR)
        #temp_paper3 = expand_1_hop_graph(about_edge_index, topic_user, TOPIC)

        paper_user.update(temp_paper1)
        paper_user.update(temp_paper2)
        #paper_user.update(temp_paper3)
        author_user.update(temp_author)
        topic_user.update(temp_topic)

        paper_user.difference_update(paper_user_scanned)
        author_user.difference_update(author_user_scanned)
        topic_user.difference_update(topic_user_scanned)

        paper_seed.difference_update(paper_user)
        author_seed.difference_update(author_user)
        topic_seed.difference_update(topic_user)



def main():
    fold = 1
    max_year = 2019
    keep_edges = False
    root = "ANP_DATA"

    dataset = ANPDataset(root=root)
    dataset[0].to('cuda:1')
    dataloader = ANPDataLoader(dataset, root=root, fold=fold, max_year=max_year, keep_edges=keep_edges)

    dataiter = iter(dataloader)
    sub_graph, sub_graph_next_year, history_author_list, papers_next_year = next(dataiter)

    sub_graph.to('cuda:1')
    sub_graph_next_year.to('cuda:1')

    tensor_paper_next_year = torch.tensor(papers_next_year).to('cuda:1')

    time = datetime.now()
    for i, author in enumerate(history_author_list):
        #if author == 74662:
        time = datetime.now()
        get_history_infosphere(sub_graph, sub_graph_next_year, i, tensor_paper_next_year)
        print(f"Infosphere creation time: {str(datetime.now() - time)}")
    print(f"History & Infosphere creation time for a fold: {str(datetime.now() - time)}")


if __name__ == "__main__":
    #cProfile.run('main()')
    main()
