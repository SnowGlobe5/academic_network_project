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


def compare_frontier(user, seed):
    for t in [PAPER, AUTHOR, TOPIC]:
        for p, frontier in seed[t]:
            if p in user[t]:
                p = None

def get_history_infosphere(data, data_next_year, author_id, papers_next_year):
    author_papers_next_year = get_papers_per_author_year(data_next_year, author_id, papers_next_year)
    writes_edge_index_next_year = data_next_year['author', 'writes', 'paper'].edge_index
    cites_edge_index_next_year = data_next_year['paper', 'cites', 'paper'].edge_index
    about_edge_index_next_year = data_next_year['paper', 'about', 'topic'].edge_index

    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index

    frontier_user = [set(), set(), set()]
    infosphere_starting_point = [set(), set(), set()]
    frontier_seeds = [{}, {}, {}]
    user_scanned = [set(), set(), set()]
    scanned_infosphere = {}

    ## History
    sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author_id, flow='target_to_source')
    author_papers = sub_edge_index[1].tolist()
    history = {('author', 'writes', 'paper'): sub_edge_index,
                ('paper', 'cites', 'paper'): torch.tensor([]).to(torch.int64).to('cuda:1'),
                ('paper', 'about', 'topic'): torch.tensor([]).to(torch.int64).to('cuda:1')}

    for paper in author_papers:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        history['paper', 'cites', 'paper'] = torch.cat((history['paper', 'cites', 'paper'], sub_edge_index), dim=1)
        for cited_paper in sub_edge_index[1].tolist():
            frontier_user[PAPER].add(cited_paper)
        
        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        history['author', 'writes', 'paper'] = torch.cat((history['author', 'writes', 'paper'], sub_edge_index[:, mask]), dim=1)
        for co_author in sub_edge_index[:, mask][0].tolist():
             frontier_user[AUTHOR].add(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        history['paper', 'about', 'topic'] = torch.cat((history['paper', 'about', 'topic'], sub_edge_index), dim=1)
        for topic in sub_edge_index[1].tolist():
            frontier_user[TOPIC].add(topic)

    ## Infosphere starting point
    for paper in author_papers_next_year:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index_next_year, paper, flow='target_to_source')
        mask = torch.isin(sub_edge_index[1], papers_next_year, invert=True)
        for cited_paper in sub_edge_index[:, mask][1].tolist():
            infosphere_starting_point[PAPER].add(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index_next_year, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            infosphere_starting_point[AUTHOR].add(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index_next_year, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            infosphere_starting_point[TOPIC].add(topic)

    
    frontier_seeds[PAPER] = [(paper, [paper]) for paper in infosphere_starting_point[PAPER]]
    frontier_seeds[AUTHOR] = [(author, [author]) for author in infosphere_starting_point[AUTHOR]]
    frontier_seeds[TOPIC] = [(topic, [topic]) for topic in infosphere_starting_point[TOPIC]]

    user_scanned[PAPER].update(frontier_user[PAPER])
    user_scanned[AUTHOR].update(frontier_user[AUTHOR])
    user_scanned[TOPIC].update(frontier_user[TOPIC])
    
    compare_frontier(frontier_user, frontier_seeds)
    
    # while (seeds not empty)
    for i in range(1):
        if not (infosphere_starting_point[PAPER] or infosphere_starting_point[AUTHOR] or infosphere_starting_point[TOPIC]):
            break
        
        # frontier-seed-new=[]
        frontier_seed_new = [set(), set(), set()]
        # for p,frontier in frontier-seeds{
        for t in [PAPER, AUTHOR, TOPIC]:
            for p, frontier in frontier_seeds[t]:
                # scanned-infosphere[p].append(frontier) 
                if scanned_infosphere.get(p):
                    scanned_infosphere[p][t].update(frontier_seeds[t])
                else:
                    scanned_infosphere[p] = [set(), set(), set()]
                
                # frontier_temp=[p,expansion of (frontier) - scanned-infosphere[p])
                (temp_paper1, temp_author, temp_topic) = expand_1_hop_graph((cites_edge_index, writes_edge_index, about_edge_index), infosphere_starting_point[PAPER], PAPER)
                temp_paper2 = expand_1_hop_graph(writes_edge_index, infosphere_starting_point[AUTHOR], AUTHOR)
                # temp_paper3 = expand_1_hop_graph(about_edge_index, topic_seed, TOPIC)
                frontier_seeds[PAPER].update(temp_paper1)
                frontier_seeds[PAPER].update(temp_paper2)
                # frontier_seeds.update(temp_paper3)
                frontier_seeds[AUTHOR].update(temp_author)
                frontier_seeds[TOPIC].update(temp_topic)

                frontier_seed_new[PAPER].update(frontier_seeds[PAPER].difference(scanned_infosphere[p][PAPER]))
                frontier_seed_new[AUTHOR].update(frontier_seeds[AUTHOR].difference(scanned_infosphere[p][AUTHOR]))
                frontier_seed_new[TOPIC].update(frontier_seeds[TOPIC].difference(scanned_infosphere[p][TOPIC]))

                
    
        if not (infosphere_starting_point[PAPER] or infosphere_starting_point[AUTHOR] or infosphere_starting_point[TOPIC]):
            break

        # append frontier-user to scanned-user    
        user_scanned[PAPER].update(frontier_user[PAPER])
        user_scanned[AUTHOR].update(frontier_user[AUTHOR])
        user_scanned[TOPIC].update(frontier_user[TOPIC])
        (temp_paper1, temp_author, temp_topic) = expand_1_hop_graph((cites_edge_index, writes_edge_index, about_edge_index), frontier_user[PAPER], PAPER)
        temp_paper2 = expand_1_hop_graph(writes_edge_index, frontier_user[AUTHOR], AUTHOR)
        #temp_paper3 = expand_1_hop_graph(about_edge_index, user[TOPIC], TOPIC)
        #
        frontier_user[PAPER].update(temp_paper1)
        frontier_user[PAPER].update(temp_paper2)
        #user[PAPER].update(temp_paper3)
        frontier_user[AUTHOR].update(temp_author)
        frontier_user[TOPIC].update(temp_topic)
        #
        frontier_user[PAPER].difference_update(user_scanned[PAPER])
        frontier_user[AUTHOR].difference_update(user_scanned[AUTHOR])
        frontier_user[TOPIC].difference_update(user_scanned[TOPIC])

        # compare frontier with user frontier and remove p if interesction not null
        compare_frontier(frontier_user, frontier_seeds)



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
        if author == 74662:
            time = datetime.now()
            get_history_infosphere(sub_graph, sub_graph_next_year, i, tensor_paper_next_year)
            print(f"Infosphere creation time: {str(datetime.now() - time)}")
    print(f"History & Infosphere creation time for a fold: {str(datetime.now() - time)}")


if __name__ == "__main__":
    #cProfile.run('main()')
    main()
