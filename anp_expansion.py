import torch
import random

# once the paths have been computed  you can  create the actual infosphere by using an extension policy. the base  policy would start from the pahts and choose randomly
# a node and randomly expand a few children. No need to check for duplicates. the only parameters would be how many nodes to expand and the maximum children node to expand.

# please keep node selection and children expansions part of the policy separated as they can be later specialised by learning or by taking into account domain specific
# heuristics like open the most cited papers etc

def random_selection_policy(node_list):
    node_type = random.randrange(3)
    list_selected = node_list[node_type]
    element_index = random.randrange(len(list_selected))
    return list_selected[element_index]

def random_expansion_policy(node, n_children, edge_list):
    # edge_list = torch.cat((infosphere_edge_list[ABOUT], element[1]), dim=1) 
    pass

# Check node_list is not empty before call expand_infosphere
def expand_infosphere(node_list, edge_list, n_node, n_children, selection_policy, expansion_policy):
    if node_list == [[], [], []]: 
        raise Exception("Not possible to expand an empty infosphere.")
    
    count = 0
    while count == n_node:
        node = selection_policy(node_list)
        if expansion_policy(node, n_children, edge_list):
            count += 1
