import torch
import anp_expansion
from anp_dataset import ANPDataset
from anp_utils import *

N_NODE = 0
N_CHILDREN = 0


def json_to_edge_list(infosphere, paper_limit, device):
    infosphere_edge_list = []
    for author_infosphere in infosphere:
        author_infosphere_edge_list = [
            torch.tensor([[],[]]).to(torch.int64).to(device),
            torch.tensor([[],[]]).to(torch.int64).to(device),
            torch.tensor([[],[]]).to(torch.int64).to(device)]
        for path in author_infosphere:
            for element in path:
                match element[0]:
                    case 'cites':
                        if element[1][0] <= paper_limit or element[1][1] <= paper_limit:
                            author_infosphere_edge_list[CITES] = torch.cat((author_infosphere_edge_list[CITES], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(device)), dim=1)
                    case 'writes':
                        if element[1][1] <= paper_limit:
                                author_infosphere_edge_list[WRITES] = torch.cat((author_infosphere_edge_list[WRITES], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(device)), dim=1)
                    case 'about':
                        if element[1][0] <= paper_limit:
                            author_infosphere_edge_list[ABOUT] = torch.cat((author_infosphere_edge_list[ABOUT], torch.Tensor([[element[1][0]],[element[1][1]]]).to(torch.int64).to(device)), dim=1)
        infosphere_edge_list.append(author_infosphere_edge_list)
    return infosphere_edge_list


def finalize_infosphere(fold, year, keep_edges, p1, p2, p3, f, limits, split, cuda):
    DEVICE=torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    authors_infosphere_edge_list = [
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE),
            torch.tensor([[],[]]).to(torch.int64).to(DEVICE)]
    fold_string = [str(x) for x in fold]
    fold_string = '_'.join(fold_string)
    root = "ANP_DATA"

    dataset = ANPDataset(root=root)
    data = dataset[0]
    sub_graph, _, _, _ = anp_filter_data(data, root=root, folds=fold, max_year=year, keep_edges=keep_edges)
    sub_graph = sub_graph.to(DEVICE)
    
    anp_expansion.color_tracker = [
        [0] * sub_graph['paper'].num_nodes,
        [0] * sub_graph['author'].num_nodes,
        [0] * sub_graph['topic'].num_nodes
    ]

    anp_expansion.exploration_tracker = [
        [0] * sub_graph['paper'].num_nodes,
        [0] * sub_graph['author'].num_nodes,
        [0] * sub_graph['topic'].num_nodes
    ]

    try:
        part = limits[0]
        while part <= limits[1]:
            print(f"Part {part}")
            with open(f"ANP_DATA/computed_infosphere/infosphere_{fold_string}_{year}_{part}.json", 'r') as json_file:
                part_dict_infosphere =  json.load(json_file)
                if split is not None:
                    half_info = int(len(part_dict_infosphere) / 2)
                    if split == 1:
                        part_dict_infosphere = part_dict_infosphere[0:half_info]
                    else:
                        part_dict_infosphere = part_dict_infosphere[half_info:len(part_dict_infosphere)]
                        
                part_tensor_infosphere = json_to_edge_list(part_dict_infosphere, sub_graph['paper'].num_nodes, DEVICE)
                
                time = datetime.now()
                tot = len(part_tensor_infosphere)
                
                for i, author_edge_list in enumerate(part_tensor_infosphere):
                    if not i % 10000:
                        print(f"author edge processed: {i}/{tot} - {i/tot*100}% - {str(datetime.now() - time)}")
                    
                    num_seeds = len(part_dict_infosphere[i])
                    
                    if part_dict_infosphere[i]:                       
                        expansion = anp_expansion.infosphere_noisy_expansion(sub_graph, author_edge_list, p1, p2, p3, f, num_seeds, part_dict_infosphere[i][0][-1][1][0], DEVICE)

                        if expansion:
                        # if anp_expansion.expand_infosphere(sub_graph, author_edge_list, N_NODE, N_CHILDREN, anp_expansion.random_selection_policy, anp_expansion.random_expansion_policy):
                            authors_infosphere_edge_list[CITES] = torch.cat((authors_infosphere_edge_list[CITES], author_edge_list[CITES]), dim=1)
                            authors_infosphere_edge_list[WRITES] = torch.cat((authors_infosphere_edge_list[WRITES], author_edge_list[WRITES]), dim=1)
                            authors_infosphere_edge_list[ABOUT] = torch.cat((authors_infosphere_edge_list[ABOUT], author_edge_list[ABOUT]), dim=1)
                            authors_infosphere_edge_list[CITES] = torch.cat((authors_infosphere_edge_list[CITES], expansion[CITES]), dim=1)
                            authors_infosphere_edge_list[WRITES] = torch.cat((authors_infosphere_edge_list[WRITES], expansion[WRITES]), dim=1)
                            authors_infosphere_edge_list[ABOUT] = torch.cat((authors_infosphere_edge_list[ABOUT], expansion[ABOUT]), dim=1)
                        else:
                            continue
                    else:
                        continue
            part += 1
            
        torch.save(authors_infosphere_edge_list, f"ANP_DATA/computed_infosphere/infosphere_{fold_string}_{year}_noisy_{limits[0]}_{limits[1]}_{split}.pt")
                    
                
    except FileNotFoundError:
        torch.save(authors_infosphere_edge_list, f"ANP_DATA/computed_infosphere/infosphere_{fold_string}_{year}_noisy_{limits[0]}_{limits[1]}_{split}.pt")
                

#finalize_infosphere([0, 1, 2, 3, 4], 2019, True, 0.5, 0.5, 0.5, 2, [0, 0], 1, 0)