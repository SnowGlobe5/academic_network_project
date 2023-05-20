def get_infosphere_old(data, data_next_year, author_id, paths, papers_next_year, last_expanded, max_expansion):
    author_papers_next_year = get_papers_per_author_year(data_next_year, author_id, papers_next_year)
    # FFprint(author_papers_next_year)
    expansion = [2, 2, 2]
    last_expanded = [last_expanded, [[], [], []]]
    writes_edge_index = data_next_year['author', 'writes', 'paper'].edge_index
    cites_edge_index = data_next_year['paper', 'cites', 'paper'].edge_index
    about_edge_index = data_next_year['paper', 'about', 'topic'].edge_index
    infosphere = {PAPER: {}, AUTHOR: {}, TOPIC: {}}
    for paper in author_papers_next_year:
        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        mask = torch.isin(sub_edge_index[1], torch.tensor(papers_next_year).to('cuda:0'), invert=True)
        for cited_paper in sub_edge_index[:, mask][1].tolist():
            if not infosphere[PAPER].get(cited_paper):
                while True:
                    if paths[PAPER].get(cited_paper):
                        infosphere[PAPER][cited_paper] = paths[PAPER][cited_paper]
                        break
                    else:
                        if expansion[0] < max_expansion:
                            lazy_expand_graph(data, paths, last_expanded, PAPER, expansion)
                        else:
                            break

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        for co_author in sub_edge_index[:, mask][0].tolist():
            if not infosphere[PAPER].get(co_author):
                while True:
                    if paths[AUTHOR].get(co_author):
                        infosphere[AUTHOR][co_author] = paths[AUTHOR][co_author]
                        break
                    else:
                        if expansion[1] < max_expansion:
                            lazy_expand_graph(data, paths, last_expanded, AUTHOR, expansion)
                        else:
                            break

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            if not infosphere[PAPER].get(topic):
                while True:
                    if paths[TOPIC].get(topic):
                        infosphere[TOPIC][topic] = paths[TOPIC][topic]
                        break
                    else:
                        if expansion[2] < max_expansion:
                            lazy_expand_graph(data, paths, last_expanded, TOPIC, expansion)
                        else:
                            break


def expand_1_hop_graph_old(data, paths, last_expanded, list_node_to_expand):
    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index
    new_last_expanded = [[], [], []]

    for paper in last_expanded[PAPER]:
        if PAPER in list_node_to_expand:
            sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
            for cited_paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(cited_paper):
                    paths[PAPER][cited_paper] = paths[PAPER][paper] + [('cites', [paper, cited_paper])]
                    new_last_expanded[PAPER].append(cited_paper)

        # co-authors
        if AUTHOR in list_node_to_expand:
            sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
            for co_author in sub_edge_index[0].tolist():
                if not paths[AUTHOR].get(co_author):
                    paths[AUTHOR][co_author] = paths[PAPER][paper] + [('writes', [co_author, paper])]
                    new_last_expanded[AUTHOR].append(co_author)

        # topic
        if TOPIC in list_node_to_expand:
            sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if not paths[TOPIC].get(topic):
                    paths[TOPIC][topic] = paths[PAPER][paper] + [('about', [paper, topic])]
                    new_last_expanded[TOPIC].append(topic)

    for author in last_expanded[AUTHOR]:
        if PAPER in list_node_to_expand:
            sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author, flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if not paths[PAPER].get(paper):
                    paths[PAPER][paper] = paths[AUTHOR][author] + [('writes', [author, paper])]
                    new_last_expanded[PAPER].append(paper)

    for topic in last_expanded[TOPIC]:
        if PAPER in list_node_to_expand:
            sub_edge_index = expand_1_hop_edge_index(about_edge_index, topic, flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if not paths[PAPER].get(paper):
                    paths[PAPER][paper] = paths[TOPIC][topic] + [('about', [paper, topic])]
                    new_last_expanded[PAPER].append(paper)

    return new_last_expanded

    def lazy_expand_graph(data, paths, last_expanded, node, expansion):
    if not last_expanded[1][node]:
        last_expanded[1][node] = expand_1_hop_graph(data, paths, last_expanded[0], [node])[node]
        expansion[node] += 1
    else:
        list_node_to_expand = []
        for nnode in [PAPER, AUTHOR, TOPIC]:
            if not last_expanded[1][nnode]:
                list_node_to_expand.append(nnode)
                expansion[nnode] += 1
        last_expanded[0] = expand_1_hop_graph(data, paths, last_expanded[0], list_node_to_expand)
        for nnode in [PAPER, AUTHOR, TOPIC]:
            if nnode not in list_node_to_expand:
                last_expanded[0][nnode] = last_expanded[1][nnode]
        last_expanded[1] = [[], [], []]
        lazy_expand_graph(data, paths, last_expanded, node, expansion)


def create_history_graph_old(data, author_id):
    paths = {
        PAPER: {},
        AUTHOR: {},
        TOPIC: {}
    }
    last_expanded = [[], [], []]
    writes_edge_index = data['author', 'writes', 'paper'].edge_index
    cites_edge_index = data['paper', 'cites', 'paper'].edge_index
    about_edge_index = data['paper', 'about', 'topic'].edge_index

    sub_edge_index = expand_1_hop_edge_index(writes_edge_index, author_id, flow='target_to_source')
    author_papers = sub_edge_index[1]
    subset_dict = {('author', 'writes', 'paper'): sub_edge_index,
                   ('paper', 'cites', 'paper'): torch.tensor([]).to(torch.int64).to('cuda:0'),
                   ('paper', 'about', 'topic'): torch.tensor([]).to(torch.int64).to('cuda:0')}
    paths[AUTHOR][author_id] = []

    for paper in author_papers.tolist():
        # add to paths 4 easy access in infosphere creation
        paths[PAPER][paper] = [('writes', [author_id, paper])]

        # cited papers
        sub_edge_index = expand_1_hop_edge_index(cites_edge_index, paper, flow='target_to_source')
        subset_dict['paper', 'cites', 'paper'] = torch.cat((subset_dict['paper', 'cites', 'paper'], sub_edge_index),
                                                           dim=1)
        for cited_paper in sub_edge_index[1].tolist():
            if not paths[PAPER].get(cited_paper):
                paths[PAPER][cited_paper] = [('writes', [author_id, paper]), ('cites', [paper, cited_paper])]
                last_expanded[PAPER].append(cited_paper)

        # co-authors
        sub_edge_index = expand_1_hop_edge_index(writes_edge_index, paper, flow='source_to_target')
        mask = sub_edge_index[0] != author_id
        subset_dict['author', 'writes', 'paper'] = torch.cat(
            (subset_dict['author', 'writes', 'paper'], sub_edge_index[:, mask]), dim=1)
        for co_author in sub_edge_index[:, mask][0].tolist():
            if not paths[AUTHOR].get(co_author):
                paths[AUTHOR][co_author] = [('writes', [author_id, paper]), ('writes', [co_author, paper])]
                last_expanded[AUTHOR].append(co_author)

        # topic
        sub_edge_index = expand_1_hop_edge_index(about_edge_index, paper, flow='target_to_source')
        subset_dict['paper', 'about', 'topic'] = torch.cat((subset_dict['paper', 'about', 'topic'], sub_edge_index),
                                                           dim=1)
        for topic in sub_edge_index[1].tolist():
            if not paths[TOPIC].get(topic):
                paths[TOPIC][topic] = [('writes', [author_id, paper]), ('about', [paper, topic])]
                last_expanded[TOPIC].append(topic)

    return subset_dict, paths, last_expanded