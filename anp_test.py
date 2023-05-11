from datetime import datetime

import torch
from torch_geometric.utils import k_hop_subgraph

from anp_dataloader import ANPDataLoader
from anp_dataset import ANPDataset


def get_papers_per_author_year(data, author, papers_year_list):
    edge_index = data['author', 'writes', 'paper'].edge_index
    sub_edge_index = expand_1_hop_edge_index(edge_index, author, flow='target_to_source')
    mask = torch.isin(sub_edge_index[1], torch.tensor(papers_year_list))
    return sub_edge_index[:, mask][1].tolist()


def expand_1_hop_edge_index(edge_index, node, flow):
    _, sub_edge_index, _, _ = k_hop_subgraph(node, 1, edge_index, flow=flow)
    # Clean
    if flow == 'target_to_source':
        mask = sub_edge_index[0] == node
    else:
        mask = sub_edge_index[1] == node
    return sub_edge_index[:, mask]


def expand_1_hop_graph(data, paths, last_expanded):
    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index
    new_last_expanded = {'paper': [], 'author': [], 'topic': []}

    for paper in last_expanded['paper']:
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        for cited_paper in sub_edge_index[1].tolist():
            if not paths['paper'].get(cited_paper):
                paths['paper'][cited_paper] = paths['paper'][paper].append(('cites', [paper, cited_paper]))
                new_last_expanded['paper'].append(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        for co_author in sub_edge_index[0].tolist():
            if not paths['author'].get(co_author):
                paths['author'][co_author] = paths['paper'][paper].append('writes', [co_author, paper])
                new_last_expanded['author'].append(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            if not paths['topic'].get(topic):
                paths['topic'][topic] = paths['paper'][paper].append('about', [paper, topic])
                new_last_expanded['topic'].append(topic)

    for author in last_expanded['author']:
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author, flow='target_to_source')
        for paper in sub_edge_index[1].tolist():
            if not paths['paper'].get(paper):
                paths['paper'][paper] = paths['author'][author].append(('writes', [author, paper]))
                new_last_expanded['paper'].append(paper)

    for topic in last_expanded['topic']:
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, topic, flow='source_to_target')
        for paper in sub_edge_index[0].tolist():
            if not paths['paper'].get(paper):
                paths['paper'][paper] = paths['topic'][topic].append(('about', [paper, topic]))
                new_last_expanded['paper'].append(paper)


def create_history_graph(data, author_id):
    paths = {
        'paper': {},
        'author': {},
        'topic': {}
    }
    last_expanded = {'paper': [], 'author': [], 'topic': []}
    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index

    sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author_id, flow='target_to_source')
    author_papers = sub_edge_index[1]
    subset_dict = {('author', 'writes', 'paper'): sub_edge_index,
                   ('paper', 'cites', 'paper'): torch.tensor([]).to(torch.int64),
                   ('paper', 'about', 'topic'): torch.tensor([]).to(torch.int64)}
    paths['author'][author_id] = []

    for paper in author_papers.tolist():
        # add to paths 4 easy access in infosphere creation
        paths['paper'][paper] = [('writes', [author_id, paper])]

        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        subset_dict['paper', 'cites', 'paper'] = torch.cat((subset_dict['paper', 'cites', 'paper'], sub_edge_index),
                                                           dim=1)
        for cited_paper in sub_edge_index[1].tolist():
            if not paths['paper'].get(cited_paper):
                paths['paper'][cited_paper] = [('writes', [author_id, paper]), ('cites', [paper, cited_paper])]
                last_expanded['paper'].append(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        subset_dict['author', 'writes', 'paper'] = torch.cat(
            (subset_dict['author', 'writes', 'paper'], sub_edge_index[:, mask]), dim=1)
        for co_author in sub_edge_index[:, mask][0].tolist():
            if not paths['author'].get(co_author):
                paths['author'][co_author] = [('writes', [author_id, paper]), ('writes', [co_author, paper])]
                last_expanded['author'].append(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        subset_dict['paper', 'about', 'topic'] = torch.cat((subset_dict['paper', 'about', 'topic'], sub_edge_index),
                                                           dim=1)
        for topic in sub_edge_index[1].tolist():
            if not paths['topic'].get(topic):
                paths['topic'][topic] = [('writes', [author_id, paper]), ('about', [paper, topic])]
                last_expanded['topic'].append(topic)

    return subset_dict, paths, last_expanded


def get_infosphere(data, data_next_year, author_id, paths, papers_next_year, last_expanded, max_expansion):
    author_papers_next_year = get_papers_per_author_year(data_next_year, author_id, papers_next_year)
    print(author_papers_next_year)
    expansion = 2
    writes_edge_index = data_next_year['author', 'writes', 'paper'].edge_index
    cites_edge_index = data_next_year['paper', 'cites', 'paper'].edge_index
    about_edge_index = data_next_year['paper', 'about', 'topic'].edge_index
    infosphere = {
        'paper': {},
        'author': {},
        'topic': {}
    }
    for paper in author_papers_next_year:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        mask = torch.isin(sub_edge_index[1], torch.tensor(papers_next_year), invert=True)
        for cited_paper in sub_edge_index[:, mask][1].tolist():
            if not infosphere['paper'].get(cited_paper):
                while expansion < max_expansion:
                    if paths['paper'].get(cited_paper):
                        infosphere['paper'][cited_paper] = paths['paper'][cited_paper]
                        break
                    else:
                        expand_1_hop_graph(data, paths, last_expanded)
                        expansion += 1

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            if not infosphere['paper'].get(co_author):
                while expansion < max_expansion:
                    if paths['author'].get(co_author):
                        infosphere['paper'][co_author] = paths['author'][co_author]
                        break
                    else:
                        expand_1_hop_graph(data, paths, last_expanded)
                        expansion += 1

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            if not infosphere['paper'].get(topic):
                while expansion < max_expansion:
                    if paths['topic'].get(topic):
                        infosphere['paper'][topic] = paths['author'][topic]
                        break
                    else:
                        expand_1_hop_graph(data, paths, last_expanded)
                        expansion += 1


def main():
    fold = 1
    max_year = 2050
    keep_edges = False
    root = "ANP_DATA"

    dataset = ANPDataset(root=root)
    dataset[0].to('cuda:0')
    dataloader = ANPDataLoader(dataset, root=root, fold=fold, max_year=max_year, keep_edges=keep_edges)

    dataiter = iter(dataloader)
    sub_graph, sub_graph_next_year, history_author_list, papers_next_year = next(dataiter)
    print(sub_graph)
    print(dataset[0])
    exit()

    # for author in history_author_list:
    time = datetime.now()
    history, paths, last_expanded = create_history_graph(sub_graph, 74662)
    print(f"History creation time: {str(datetime.now() - time)}")
    time = datetime.now()
    get_infosphere(sub_graph, sub_graph_next_year, 74662, paths, papers_next_year, last_expanded, 5)
    print(f"Infosphere creation time: {str(datetime.now() - time)}")


if __name__ == "__main__":
    main()
