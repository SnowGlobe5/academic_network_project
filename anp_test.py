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


def expand_1_hop_graph(edge_index, nodes, type, paths):
    if type == PAPER:
        paper_nodes =  []
        author_nodes = []
        topic_nodes = []
        for paper in nodes:
            # cited paper
            sub_edge_index = expand_1_hop_edge_index(edge_index[0], paper, flow='target_to_source')
            for cited_paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(cited_paper):
                    paper_nodes.append(cited_paper)
                    paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [paper, cited_paper])]

            # co-authors
            sub_edge_index = expand_1_hop_edge_index(edge_index[1], paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                if not paths[AUTHOR].get(co_author):
                    author_nodes.append(co_author)
                    paths[AUTHOR][co_author] = paths[PAPER][paper] + [('writes', [co_author, paper])]

            # topic
            sub_edge_index = expand_1_hop_edge_index(edge_index[2], paper, flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if not paths[TOPIC].get(topic):
                    topic_nodes.append(topic)
                    paths[TOPIC][topic] = paths[PAPER][paper] + [('about', [paper, topic])]

        return (paper_nodes, author_nodes, topic_nodes)

    elif type == AUTHOR:
        paper_nodes = []
        for author in nodes:
            sub_edge_index = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[AUTHOR][author] + [('writes', [author, paper])]
        return paper_nodes

    else:
        paper_nodes = []
        for topic in nodes:
            sub_edge_index = expand_1_hop_edge_index(edge_index, topic, flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(paper):
                    paper_nodes.append(paper)
                    paths[PAPER][paper] = paths[TOPIC][topic] + [('about', [paper, topic])]
        return paper_nodes
  

def get_papers_per_author_year(data, author, papers_year_list):
    edge_index = data['author', 'writes', 'paper'].edge_index
    sub_edge_index = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
    mask = torch.isin(sub_edge_index[1], papers_year_list)
    return sub_edge_index[:, mask][1]


def compare_frontier(user, seeds, infosphere, is_expand_seed):
    #time = datetime.now()
    key_to_delete = set()
    for t in [PAPER, AUTHOR, TOPIC]:
        # Questi sono i seed trovati all'inizio
        for i, seed in seeds.items():
            # La loro frontiera attuale
            if seed == [[], [], []]:
                # Non porta da nessuna parte, non possiamo collegarlo a nulla
                key_to_delete.add(i)
                break
            if is_expand_seed:
                for node in seed[t]:
                    if user[t].get(node):
                        # Add to global infosphere
                        key_to_delete.add(i)
                        break
            else:
                for node in user[t]:
                    if infosphere[i][t].get(node):
                        # Add to global infosphere
                        key_to_delete.add(i)
                        break
    for key in key_to_delete:
        del seeds[key] # Rimuovi tutto il seed
        del infosphere[key]
    #print(f"Compare frontier time: {datetime.now()-time}")
    return seeds


def update_scanned_expand(scanned, frontier, cites_edge_index, writes_edge_index, about_edge_index):
    # for p,frontier in frontier-seeds{
    (temp_paper1, temp_author, temp_topic) = expand_1_hop_graph((cites_edge_index, writes_edge_index, about_edge_index), frontier[PAPER], PAPER, scanned)
    temp_paper2 = expand_1_hop_graph(writes_edge_index, frontier[AUTHOR], AUTHOR, scanned)
    temp_paper3 = expand_1_hop_graph(about_edge_index, frontier[TOPIC], TOPIC, scanned)
    #
    frontier = [[], [], []]
    frontier[PAPER].extend(temp_paper1)
    frontier[PAPER].extend(temp_paper2)
    frontier[PAPER].extend(temp_paper3)
    frontier[AUTHOR].extend(temp_author)
    frontier[TOPIC].extend(temp_topic)
    return frontier


def update_scanned_expand_seeds(scanned, frontier, cites_edge_index, writes_edge_index, about_edge_index):
    new_frontier = {}
    for seed in frontier.keys():
        new_frontier[seed] = update_scanned_expand(scanned[seed], frontier[seed], cites_edge_index, writes_edge_index, about_edge_index)
    return new_frontier


def get_history_infosphere(data, data_next_year, author_id, papers_next_year):
    author_papers_next_year = get_papers_per_author_year(data_next_year, author_id, papers_next_year)
    writes_edge_index_next_year = data_next_year['author', 'writes', 'paper'].edge_index
    cites_edge_index_next_year = data_next_year['paper', 'cites', 'paper'].edge_index
    about_edge_index_next_year = data_next_year['paper', 'about', 'topic'].edge_index

    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index

    infosphere_starting_point = [set(), set(), set()]

    frontier_user = [[], [], []]
    scanned_user = [{}, {}, {}]
    frontier_seeds = {}
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
            if not scanned_user[PAPER].get(cited_paper):
                frontier_user[PAPER].append(cited_paper)
                scanned_user[PAPER][cited_paper] = [('writes', [author_id, paper]), ('cites', [paper, cited_paper])]
        
        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        history['author', 'writes', 'paper'] = torch.cat((history['author', 'writes', 'paper'], sub_edge_index[:, mask]), dim=1)
        for co_author in sub_edge_index[:, mask][0].tolist():
            if not scanned_user[AUTHOR].get(co_author):
                frontier_user[AUTHOR].append(co_author)
                scanned_user[AUTHOR][co_author] = [('writes', [author_id, paper]), ('writes', [co_author, paper])]

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        history['paper', 'about', 'topic'] = torch.cat((history['paper', 'about', 'topic'], sub_edge_index), dim=1)
        for topic in sub_edge_index[1].tolist():
            if not scanned_user[TOPIC].get(topic):
                frontier_user[TOPIC].append(topic)
                scanned_user[TOPIC][topic] = [('writes', [author_id, paper]), ('about', [paper, topic])]

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

    for paper in infosphere_starting_point[PAPER]:
        frontier_seeds[('paper', paper)] = [[], [], []]
        frontier_seeds[('paper', paper)][PAPER] = [paper]
        scanned_infosphere[('paper', paper)] = [{}, {}, {}]
        scanned_infosphere[('paper', paper)][PAPER][paper] = []

    for author in infosphere_starting_point[AUTHOR]: 
        frontier_seeds[('author', author)] = [[], [], []]
        frontier_seeds[('author', author)][AUTHOR] = [author]
        scanned_infosphere[('author', author)] = [{}, {}, {}]
        scanned_infosphere[('author', author)][AUTHOR][author] = []

    for topic in infosphere_starting_point[TOPIC]:
        frontier_seeds[('topic', topic)] = [[], [], []]
        frontier_seeds[('topic', topic)][TOPIC] = [topic]
        scanned_infosphere[('topic', topic)] = [{}, {}, {}]
        scanned_infosphere[('topic', topic)][TOPIC][topic] = []
    
    total_seed = len(frontier_seeds)
    if compare_frontier(scanned_user, frontier_seeds, scanned_infosphere, True):
        # while (seeds not empty)
        while True:            
            # Expand seed frontier
            frontier_seeds = update_scanned_expand_seeds(scanned_infosphere, frontier_seeds, cites_edge_index, writes_edge_index, about_edge_index)
            if not compare_frontier(scanned_user, frontier_seeds, scanned_infosphere, True): break

            # append frontier-user to scanned-user    
            frontier_user = update_scanned_expand(scanned_user, frontier_user, cites_edge_index, writes_edge_index, about_edge_index)
            if not compare_frontier(frontier_user, frontier_seeds, scanned_infosphere, False): break

    return len(frontier_seeds)/total_seed


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

    time_total = datetime.now()
    maxt = 0
    seed_drop = 0
    for i, author in enumerate(history_author_list):
        #if author == 74662:
        if i == 1000:
            break
        time = datetime.now()
        seed_drop += get_history_infosphere(sub_graph, sub_graph_next_year, i, tensor_paper_next_year)
        post_time = str(datetime.now() - time)
        if post_time > maxt: maxt = post_time
        # print(f"Infosphere creation time: {str(datetime.now() - time)}")
    print(f"History & Infosphere creation time for a fold: {str(datetime.now() - time)}")


if __name__ == "__main__":
    #cProfile.run('main()')
    main()
