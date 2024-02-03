import json
# import cProfile
import io
import sys
import multiprocessing as mp

import torch

from anp_dataset import ANPDataset
from anp_utils import *


def get_papers_per_author_year(data, author, papers_year_list):
    edge_index = data['author', 'writes', 'paper'].edge_index
    sub_edge_index, _ = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
    mask = torch.isin(sub_edge_index[1], papers_year_list)
    return sub_edge_index[:, mask][1]


def compare_frontier(frontier_user, scanned_user, frontier_seeds, scanned_infosphere, infosphere_found, is_expand_seed):
    # time = datetime.now()
    key_to_delete = []
    for t in [PAPER, AUTHOR, TOPIC]:
        for i, seed in frontier_seeds.items():
            if seed == [[], [], []] and i not in key_to_delete:
                key_to_delete.append(i)
                break
            if is_expand_seed:
                for node in seed[t]:
                    if scanned_user[t].get(node) and i not in key_to_delete:
                        infosphere_found.append(scanned_infosphere[i][t].get(node) + scanned_user[t].get(node)[::-1])
                        key_to_delete.append(i)
                        break
            else:
                for node in frontier_user[t]:
                    if scanned_infosphere[i][t].get(node) and i not in key_to_delete:
                        infosphere_found.append(scanned_infosphere[i][t].get(node) + scanned_user[t].get(node)[::-1])
                        key_to_delete.append(i)
                        break
    for key in key_to_delete:
        del frontier_seeds[key]
        del scanned_infosphere[key]
    # print(f"Compare frontier time: {datetime.now()-time}")
    return frontier_seeds


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


def get_history_infosphere_author(data, data_next_year, author_id, papers_next_year):
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

    infosphere_found = []

    ## History
    sub_edge_index, mask = expand_1_hop_edge_index(writes_edge_index, author_id, flow='target_to_source')
    author_papers = sub_edge_index[1].tolist()
    # 0 cites, 1 writes, 2 topic
    history_mask = [torch.zeros_like(cites_edge_index[0], dtype=torch.bool),
                    torch.zeros_like(writes_edge_index[0], dtype=torch.bool),
                    torch.zeros_like(about_edge_index[0], dtype=torch.bool)]
    history_mask[WRITES] |= mask

    for paper in author_papers:
        # cited papers
        sub_edge_index, mask = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        history_mask[CITES] |= mask
        for cited_paper in sub_edge_index[1].tolist():
            if not scanned_user[PAPER].get(cited_paper):
                frontier_user[PAPER].append(cited_paper)
                scanned_user[PAPER][cited_paper] = [('writes', [author_id, paper]), ('cites', [paper, cited_paper])]

        # co-authors
        sub_edge_index, mask = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask_auth = sub_edge_index[0] != author_id
        mask &= (writes_edge_index[0] != author_id)
        history_mask[WRITES] |= mask
        for co_author in sub_edge_index[:, mask_auth][0].tolist():
            if not scanned_user[AUTHOR].get(co_author):
                frontier_user[AUTHOR].append(co_author)
                scanned_user[AUTHOR][co_author] = [('writes', [author_id, paper]), ('writes', [co_author, paper])]

        # topic
        sub_edge_index, mask = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        history_mask[ABOUT] |= mask
        for topic in sub_edge_index[1].tolist():
            if not scanned_user[TOPIC].get(topic):
                frontier_user[TOPIC].append(topic)
                scanned_user[TOPIC][topic] = [('writes', [author_id, paper]), ('about', [paper, topic])]

    ## Infosphere starting point
    for paper in author_papers_next_year:
        # cited papers
        sub_edge_index, _ = expand_1_hop_edge_index(cites_edge_index_next_year, paper, flow='target_to_source')
        mask = torch.isin(sub_edge_index[1], papers_next_year, invert=True)
        for cited_paper in sub_edge_index[:, mask][1].tolist():
            infosphere_starting_point[PAPER].add(cited_paper)

        # co-authors
        sub_edge_index, _ = expand_1_hop_edge_index(writes_edge_index_next_year, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            infosphere_starting_point[AUTHOR].add(co_author)

        # topic
        sub_edge_index, _ = expand_1_hop_edge_index(about_edge_index_next_year, paper, flow='target_to_source')
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

    if compare_frontier(frontier_user, scanned_user, frontier_seeds, scanned_infosphere, infosphere_found, True):
        # while (seeds not empty)
        for _ in range(MAX_ITERATION):
            # Expand seed frontier
            frontier_seeds = update_scanned_expand_seeds(scanned_infosphere, frontier_seeds, cites_edge_index, writes_edge_index, about_edge_index)
            if not compare_frontier(frontier_user, scanned_user, frontier_seeds, scanned_infosphere, infosphere_found, True): break

            # append frontier-user to scanned-user
            # frontier_user = update_scanned_expand(scanned_user, frontier_user, cites_edge_index, writes_edge_index, about_edge_index)
            # if not compare_frontier(frontier_user, scanned_user, frontier_seeds, scanned_infosphere, infosphere_found, False): break

    list_missing_seeds = list(frontier_seeds.keys())
    return history_mask, infosphere_found, list_missing_seeds


def create_mask_edge_index(edge_index, specified_edge):
    mask = torch.zeros_like(edge_index[0], dtype=torch.bool)
    try:
        for i in range(specified_edge.size(1)):
            edge = specified_edge[:, i]
            mask |= ((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))
    except IndexError:
        pass
    return mask


def save_infosphere(root, fold, max_year, infosphere, missing_seeds, part, index):
    infosphere_file = io.open(f"{root}/computed_infosphere/{max_year}/log_{fold}_{max_year}_{part}", "w", encoding="utf-8")
    infosphere_file.write(f"{index}")
    infosphere_file.close()
    
    infosphere_file = io.open(f"{root}/computed_infosphere/{max_year}/infosphere_{fold}_{max_year}_{part}.json", "w", encoding="utf-8")
    infosphere_file.write(json.dumps(infosphere))
    infosphere_file.close()

    missing_seeds_file = io.open(f"{root}/computed_infosphere/{max_year}/missing_seeds_{fold}_{max_year}_{part}.json", "w", encoding="utf-8")
    missing_seeds_file.write(json.dumps(missing_seeds))
    missing_seeds_file.close()
 
          
def generate_infosphere_part(max_year, part, start, finish):
    root = "ANP_DATA"
    
    if d_cuda:
        if part % 2 == 0: DEVICE = 'cuda:0'
        else: DEVICE = 'cuda:1'

    dataset = ANPDataset(root=root)
    data = dataset[0]
    sub_graph, sub_graph_next_year, history_author_list, papers_next_year = anp_filter_data(data, root=root, folds=n_fold, max_year=year, keep_edges=keep_edges)

    sub_graph = sub_graph.to(DEVICE)
    sub_graph_next_year = sub_graph_next_year.to(DEVICE)

    tensor_paper_next_year = torch.tensor(papers_next_year).to(DEVICE)
    #history_mask = []
    infosphere = []
    missing_seeds = []

    time = datetime.now()
    tot = finish - start
    #for i, author in enumerate(history_author_list):
    for i, author in enumerate(range(start, finish)):
        if i % 100 == 1:
            save_infosphere(root, fold_string, max_year, infosphere, missing_seeds, part, author)
            delta = datetime.now() - time
            remaining = tot * delta / i - delta
            print(f"part {part}) author processed: {i}/{tot} - {i/tot*100}% - elapsed: {str(delta)} - remaining: {remaining}")
        _, author_infosphere, author_missing_seeds = \
            get_history_infosphere_author(sub_graph, sub_graph_next_year, author, tensor_paper_next_year)
        infosphere.append(author_infosphere)
        missing_seeds.append(author_missing_seeds)
    
    save_infosphere(root, fold_string, max_year, infosphere, missing_seeds, part, finish)
            

def main():
    tot = 5259857  #TODO 
    delta = (int) (tot / n_parts)
    mp.set_start_method('spawn', force=True)
    for i in range(n_parts-1):
        #generate_infosphere_part(year, i, delta*i, delta*(i+1))
        if not specific_part or specific_part == i:
            p = mp.Process(target=generate_infosphere_part, args=((year, i, delta*i, delta*(i+1))))
            p.start()
    if not specific_part or specific_part == n_parts-1:
        p = mp.Process(target=generate_infosphere_part, args=((year, n_parts-1, delta*(n_parts-1), tot)))
        p.start()


year = int(sys.argv[1])
n_fold = int(sys.argv[2])
if n_fold == -1:
    n_fold = [0, 1, 2, 3, 4]
else:
    n_fold = [n_fold]
if sys.argv[3] == 'True':
    keep_edges = True
else:
    keep_edges = False
    
n_parts = int(sys.argv[4])
if sys.argv[5] == 'True':
    d_cuda = True
else:
    d_cuda = False
fold_string = [str(x) for x in n_fold]
fold_string = '_'.join(fold_string)

specific_part = int(sys.argv[6])
if specific_part == -1:
    specific_part = None
    
if __name__ == "__main__":
    main()
