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

        
def set_color(edges, color, color_tracker):
    for element in edges[CITES]:
        if not color_tracker['paper'][element[0]]:
            color_tracker['paper'][element[0]] = color
        if not color_tracker['paper'][element[1]]:
            color_tracker['paper'][element[1]] = color
            
    for element in edges[WRITES]:
        if not color_tracker['author'][element[0]]:
            color_tracker['author'][element[0]] = color
        if not color_tracker['paper'][element[1]]:
            color_tracker['paper'][element[1]] = color
            
    for element in edges[ABOUT]:
        if not color_tracker['paper'][element[0]]:
            color_tracker['paper'][element[0]] = color
        if not color_tracker['topic'][element[1]]:
            color_tracker['topic'][element[1]] = color
    

def expand_seeds(current_node, color, data, color_tracker, exploration_tracker, expanded_edge):
    nodes = {}
    if current_node[0] == PAPER:
        sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'cites', 'paper'].edge_index, current_node[1], flow='target_to_source')
        for paper in sub_edge_index[1].tolist():
            if color_tracker['paper'][paper] == color:
                nodes[(PAPER, paper)] = exploration_tracker['paper'][paper]
                    
        sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, current_node[1], flow='source_to_target')
        for author in sub_edge_index[0].tolist():
            if color_tracker['author'][author] == color:
                nodes[(AUTHOR, author)] = exploration_tracker['author'][author]
                
        sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, current_node[1], flow='target_to_source')
        for topic in sub_edge_index[1].tolist():
            if color_tracker['topic'][topic] == color:
                nodes[(TOPIC, topic)] = exploration_tracker['topic'][topic]         
                  
    elif current_node[0] == AUTHOR:
        sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, current_node[1], flow='target_to_source')
        for paper in sub_edge_index[1].tolist():
            if color_tracker['paper'][paper] == color:
                nodes[(PAPER, paper)] = exploration_tracker['paper'][paper]
        
    elif current_node[0] == TOPIC:
        sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, current_node[1], flow='source_to_target')
        for paper in sub_edge_index[0].tolist():
            if color_tracker['paper'][paper] == color:
                nodes[(PAPER, paper)] = exploration_tracker['paper'][paper]
 
    if nodes:
        min_exploration = min(nodes.values())
        min_nodes = {k for k, v in nodes.items() if v == min_exploration}
        selected_node = random.choice(min_nodes)
        exploration_tracker[selected_node[0]][selected_node[1]] += 1
        if not color:
            color_tracker[selected_node[0]][selected_node[1]] = 'green'
            
            if current_node[0] == PAPER:
                if selected_node[0] == PAPER:
                    expanded_edge[CITES].append([current_node[1], selected_node[1]])
                elif selected_node[0] == AUTHOR:
                    expanded_edge[WRITES].append([selected_node[1], current_node[1]])
                elif selected_node[0] == TOPIC:
                    expanded_edge[ABOUT].append([current_node[1], selected_node[1]])
                    
            elif current_node[0] == AUTHOR:
                expanded_edge[WRITES].append([current_node[1], selected_node[1]])
                
            elif current_node[0] == TOPIC:
                expanded_edge[ABOUT].append([selected_node[1], current_node[1]])
        return selected_node
    else:
        return None

def infosphere_noisy_expansion(full_graph, seeds_graph, p1, p2, p3, f, num_seeds, author_node):
    expanded_edge = []
    p = {
        'orange': p1,
        'green': p2
    }
    node_to_add = num_seeds * f
    
    color_tracker = {
        'paper': [None] * full_graph['paper'].num_nodes,
        'author': [None] * full_graph['author'].num_nodes,
        'topic': [None] * full_graph['topic'].num_nodes
    }
    
    exploration_tracker = {
        'paper': [0] * full_graph['paper'].num_nodes,
        'author': [0] * full_graph['author'].num_nodes,
        'topic': [0] * full_graph['topic'].num_nodes
    }
	
    set_color(seeds_graph, 'orange', color_tracker)  # Color all seeds in the infosphere.

    current_node = (AUTHOR, author_node)
    current_color = 'orange'

    while node_to_add:
        if random.randint() > p[current_color]:
            # p = True, follow the current_color path.
            new_node = expand_seeds(current_node, current_color, full_graph, color_tracker, exploration_tracker, expanded_edge)
            if new_node:
                current_node = new_node
            else:
                # No current_color nodes available, change direction.
                new_node = expand_seeds(current_node, None, full_graph, color_tracker, exploration_tracker, expanded_edge)
                if new_node:
                    node_to_add -= 1
                    current_node = new_node
                    current_color = 'green'
                else:
                    # This is impossible, the graph is connected
                    pass
        else:
            # p = False, change direction.
            new_node = expand_seeds(current_node, None, full_graph, color_tracker, exploration_tracker, expanded_edge)
            if new_node:
                node_to_add -= 1
                current_node = new_node
                current_color = 'green'
            else:
                # No white nodes available, , follow the current_color path.
                new_node = expand_seeds(current_node, current_color, full_graph, color_tracker, exploration_tracker, expanded_edge)
                if new_node:
                    current_node = new_node
                else:
                    # This is impossible, the graph is connected
                    pass
                    
        if random.randint() > p[current_color] > p3:
            current_node = (AUTHOR, author_node)
            current_color = 'orange'
     
    return expanded_edge