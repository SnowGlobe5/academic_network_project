import torch
import random

from anp_utils import *

# once the paths have been computed  you can  create the actual infosphere by using an extension policy. the base  policy would start from the pahts and choose randomly
# a node and randomly expand a few children. No need to check for duplicates. the only parameters would be how many nodes to expand and the maximum children node to expand.

# please keep node selection and children expansions part of the policy separated as they can be later specialised by learning or by taking into account domain specific
# heuristics like open the most cited papers etc

def select_random_index(extracted_nodes):
    lenght = len(extracted_nodes)
    if lenght > 0:
        return random.randrange(lenght)

def random_selection_policy(edge_list):
    node_range = list(range(3))
    while True:
        node_type = random.choice(node_range)
        type_edge_list = edge_list[node_type]
        if node_type == 0 or type_edge_list.nelement():
            break
        else:
            node_range.remove(node_type)
    
    try:
        match node_type:
            # PAPER
            case 0:
                paper_from_range = list(range(4))
                while True:
                    paper_from = random.choice(paper_from_range)
                    if paper_from == 4: 
                        type_edge_list = edge_list[0]
                    else: 
                        type_edge_list = edge_list[paper_from]
            
                    if type_edge_list.nelement():
                        break
                    else:
                        paper_from_range.remove(node_type)
                        
                match paper_from:
                    # CITES
                    case 0:
                        list_selected = type_edge_list[0]
                        element_index = random.randrange(len(list_selected))
                    # WRITES
                    case 1:
                        list_selected = type_edge_list[1]
                        element_index = random.randrange(len(list_selected))
                    # ABOUT
                    case 2:
                        list_selected = type_edge_list[0]
                        element_index = random.randrange(len(list_selected))
                    # CITES 
                    case 3:
                        list_selected = type_edge_list[0]
                        element_index = random.randrange(len(list_selected))
                
            # AUTHOR
            case 1:
                list_selected = type_edge_list[0]
                element_index = random.randrange(len(list_selected))
            # TOPIC
            case 2: 
                list_selected = type_edge_list[1]
                element_index = random.randrange(len(list_selected))
        return list_selected[element_index], node_type
    except:
        return None, None

def random_expansion_policy(data, node, node_type, n_children, edge_list):
    for i in range(n_children):
        if node_type == PAPER:
            edge_type_to_expand = node_type_to_expand = random.randrange(3)
            
            match edge_type_to_expand:
                # CITES
                case 0:
                    sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'cites', 'paper'].edge_index, node, flow='target_to_source')
                    position = 1
                    
                # WRITES
                case 1:
                    sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, node, flow='source_to_target')
                    position = 0
                # ABOUT
                case 2:
                    sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, node, flow='target_to_source')
                    position = 1
                    
        elif node_type == AUTHOR:
            sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, node, flow='target_to_source')
            position = 1
            edge_type_to_expand = WRITES
            #node_type_to_expand = PAPER
            
        elif node_type == TOPIC:
            sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, node, flow='source_to_target')
            position = 0
            edge_type_to_expand = ABOUT
            #node_type_to_expand = PAPER
        
    index_node = select_random_index(sub_edge_index[position])
    if index_node:
        mask = sub_edge_index[position] == sub_edge_index[position][index_node]
        edge_list[edge_type_to_expand] = torch.cat((edge_list[edge_type_to_expand], sub_edge_index[:, mask]), dim=1)
        #node_list[node_type_to_expand] = torch.cat((node_list[node_type_to_expand], sub_edge_index[position][index_node]), dim=1)
        
    return True

# Check edge_list is not empty before call expand_infosphere
def expand_infosphere(data, edge_list, n_node, n_children, selection_policy, expansion_policy):
    if not edge_list[0].nelement() and not edge_list[1].nelement() and not edge_list[2].nelement(): 
        #raise Exception("Not possible to expand an empty infosphere.")
        return False
    
    count_node_expanded = 0
    #retry = 1 #TODO is it worth retying?
    while count_node_expanded < n_node: #or retry == 0:
        node, node_type = selection_policy(edge_list)
        if node:
            expansion_policy(data, node, node_type, n_children, edge_list)
        count_node_expanded += 1
            # else:
            #     retry -= 1
    return True
        
