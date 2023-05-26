import json
# import cProfile
import sys
from datetime import datetime

import torch

from anp_dataloader import ANPDataLoader
from anp_dataset import ANPDataset

PAPER = 0
AUTHOR = 1
TOPIC = 2

CITES = 0
WRITES = 1
ABOUT = 2

MAX_ITERATION = 1
DEVICE = 'cuda:0'


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
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[0], paper, flow='target_to_source')
            for cited_paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(cited_paper):
                    paper_nodes.append(cited_paper)
                    paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [paper, cited_paper])]

            # co-authors
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[1], paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                if not paths[AUTHOR].get(co_author):
                    author_nodes.append(co_author)
                    paths[AUTHOR][co_author] = paths[PAPER][paper] + [('writes', [co_author, paper])]

            # topic
            sub_edge_index, _ = expand_1_hop_edge_index(edge_index[2], paper, flow='target_to_source')
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


def main(year, n_fold, keep_relation):
    fold = n_fold
    max_year = year
    keep_edges = keep_relation
    root = "ANP_DATA"

    dataset = ANPDataset(root=root)
    dataset[0].to(DEVICE)
    dataloader = ANPDataLoader(dataset, root=root, fold=fold, max_year=max_year, keep_edges=keep_edges)

    dataiter = iter(dataloader)
    sub_graph, sub_graph_next_year, history_author_list, papers_next_year = next(dataiter)

    sub_graph.to(DEVICE)
    sub_graph_next_year.to(DEVICE)

    tensor_paper_next_year = torch.tensor(papers_next_year).to(DEVICE)

    history_mask = []
    infosphere = []
    missing_seeds = []

    for i, author in enumerate(history_author_list):
        if i == 153860:
            time = datetime.now()
            # print(sub_graph['author']['id'][i])
            author_history_mask, author_infosphere, author_missing_seeds = \
                get_history_infosphere(sub_graph, sub_graph_next_year, i, tensor_paper_next_year)
            history_mask.append(author_history_mask)
            infosphere.append(author_infosphere)
            missing_seeds.append(author_missing_seeds)

            # print(f"Infosphere creation time: {str(datetime.now() - time)}")
    torch.save(history_mask, f"output/history_{fold}_{max_year}.pt")

    infosphere_file = open(f"output/infosphere_{fold}_{max_year}.json", "w", encoding="utf-8")
    infosphere_file.write(json.dumps(infosphere))
    infosphere_file.close()

    missing_seeds_file = open(f"output/missing_seeds_{fold}_{max_year}.json", "w", encoding="utf-8")
    missing_seeds_file.write(json.dumps(missing_seeds))
    missing_seeds_file.close()

    infosphere_edge_list = []
    for author_infosphere in infosphere:
        author_infosphere_edge_list = [
            torch.tensor([]).to(torch.int64).to(DEVICE),
            torch.tensor([]).to(torch.int64).to(DEVICE),
            torch.tensor([]).to(torch.int64).to(DEVICE)]
        for element in author_infosphere:
            match element[0]:
                case 'cites':
                    author_infosphere_edge_list[0] = torch.cat((infosphere_edge_list[CITES], element[1]), dim=1)
                case 'writes':
                    author_infosphere_edge_list[1] = torch.cat((infosphere_edge_list[WRITES], element[1]), dim=1)
                case 'about':
                    author_infosphere_edge_list[2] = torch.cat((infosphere_edge_list[ABOUT], element[1]), dim=1)
        infosphere_edge_list.append(author_infosphere_edge_list)

    infosphere_mask_list = []
    for author_infosphere in infosphere_edge_list:
        infosphere_mask = [
            create_mask_edge_index(sub_graph['paper', 'cites', 'paper'].edge_index, author_infosphere[CITES]),
            create_mask_edge_index(sub_graph['author', 'writes', 'paper'].edge_index, author_infosphere[WRITES]),
            create_mask_edge_index(sub_graph['paper', 'about', 'topic'].edge_index, author_infosphere[ABOUT])]
        infosphere_mask_list.append(infosphere_mask)

    torch.save(infosphere_mask_list, f"output/infosphere_{fold}_{max_year}.pt")


if __name__ == "__main__":
    year = int(sys.argv[1])
    n_fold = int(sys.argv[2])
    if sys.argv[3] == 'True':
        keep_relation = True
    else:
        keep_relation = False
    # cProfile.run('main()')
    main(year, n_fold, keep_relation)
