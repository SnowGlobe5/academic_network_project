import torch
import random

from anp_utils import *

# once the paths have been computed  you can  create the actual infosphere by using an extension policy. the base  policy would start from the pahts and choose randomly
# a node and randomly expand a few children. No need to check for duplicates. the only parameters would be how many nodes to expand and the maximum children node to expand.

# please keep node selection and children expansions part of the policy separated as they can be later specialised by learning or by taking into account domain specific
# heuristics like open the most cited papers etc

color_tracker = []
exploration_tracker = []
WHITE = 0
ORANGE = 1
GREEN = 2
GREEN_WHITE = 3

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

        
def set_color(edges, color, color_tracker, node_list):
    for paper in edges[CITES][0]:
        if not color_tracker[PAPER][paper]:
            color_tracker[PAPER][paper] = color
            node_list[PAPER].append(paper)
    for paper in edges[CITES][1]:
        if not color_tracker[PAPER][paper]:
            color_tracker[PAPER][paper] = color
            node_list[PAPER].append(paper)
        
    for author in edges[WRITES][0]:
        if not color_tracker[AUTHOR][author]:
            color_tracker[AUTHOR][author] = color 
            node_list[AUTHOR].append(author)
    for paper in edges[WRITES][1]:
        if not color_tracker[PAPER][paper]:
            color_tracker[PAPER][paper] = color
            node_list[PAPER].append(paper)
        
    for paper in edges[ABOUT][0]:
        if not color_tracker[PAPER][paper]:
            color_tracker[PAPER][paper] = color
            node_list[PAPER].append(paper)
    for topic in edges[ABOUT][1]:     
        if not color_tracker[TOPIC][topic]:
            color_tracker[TOPIC][topic] = color
            node_list[TOPIC].append(topic)


def expand_seeds(current_node, color, data, color_tracker, exploration_tracker, seeds_edges, expanded_edges, node_list, last_node, device):
    nodes = {}
    if not color:
        if current_node[0] == PAPER:
            sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'cites', 'paper'].edge_index, current_node[1], flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]
                        
            sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, current_node[1], flow='source_to_target')
            for author in sub_edge_index[0].tolist():
                if color_tracker[AUTHOR][author] in color and (last_node[0] != AUTHOR or last_node[1] != author):
                    nodes[(AUTHOR, author)] = exploration_tracker[AUTHOR][author]
                    
            sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, current_node[1], flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if color_tracker[TOPIC][topic] in color and (last_node[0] != TOPIC or last_node[1] != topic):
                    nodes[(TOPIC, topic)] = exploration_tracker[TOPIC][topic]         
                    
        elif current_node[0] == AUTHOR:
            sub_edge_index, _ = expand_1_hop_edge_index(data['author', 'writes', 'paper'].edge_index, current_node[1], flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]
            
        elif current_node[0] == TOPIC:
            sub_edge_index, _ = expand_1_hop_edge_index(data['paper', 'about', 'topic'].edge_index, current_node[1], flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]
    else:
        if color == ORANGE:
            edges = seeds_edges
        else:
            edges = expanded_edges
            
        if current_node[0] == PAPER:
            sub_edge_index, _ = expand_1_hop_edge_index(edges[CITES], current_node[1], flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]
                        
            sub_edge_index, _ = expand_1_hop_edge_index(edges[WRITES], current_node[1], flow='source_to_target')
            for author in sub_edge_index[0].tolist():
                if color_tracker[AUTHOR][author] in color and (last_node[0] != AUTHOR or last_node[1] != author):
                    nodes[(AUTHOR, author)] = exploration_tracker[AUTHOR][author]
                    
            sub_edge_index, _ = expand_1_hop_edge_index(edges[ABOUT], current_node[1], flow='target_to_source')
            for topic in sub_edge_index[1].tolist():
                if color_tracker[TOPIC][topic] in color and (last_node[0] != TOPIC or last_node[1] != topic):
                    nodes[(TOPIC, topic)] = exploration_tracker[TOPIC][topic]         
                    
        elif current_node[0] == AUTHOR:
            sub_edge_index, _ = expand_1_hop_edge_index(edges[WRITES], current_node[1], flow='target_to_source')
            for paper in sub_edge_index[1].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]
            
        elif current_node[0] == TOPIC:
            sub_edge_index, _ = expand_1_hop_edge_index(edges[ABOUT], current_node[1], flow='source_to_target')
            for paper in sub_edge_index[0].tolist():
                if color_tracker[PAPER][paper] in color and (last_node[0] != PAPER or last_node[1] != paper):
                    nodes[(PAPER, paper)] = exploration_tracker[PAPER][paper]

 
    if nodes:
        min_exploration = min(nodes.values())
        min_nodes = [k for k, v in nodes.items() if v == min_exploration]
        selected_node = random.choice(min_nodes)
        exploration_tracker[selected_node[0]][selected_node[1]] += 1
        if not color:
            color_tracker[selected_node[0]][selected_node[1]] = GREEN
            node_list[selected_node[0]].append(selected_node[1])

            if current_node[0] == PAPER:
                if selected_node[0] == PAPER:
                    expanded_edges[CITES] = torch.cat((expanded_edges[CITES], torch.Tensor([[current_node[1]],[selected_node[1]]]).to(torch.int64).to(device)), dim=1)
                
                elif selected_node[0] == AUTHOR:
                    expanded_edges[WRITES] = torch.cat((expanded_edges[WRITES], torch.Tensor([[selected_node[1]],[current_node[1]]]).to(torch.int64).to(device)), dim=1)
            
                elif selected_node[0] == TOPIC:
                    expanded_edges[ABOUT] = torch.cat((expanded_edges[ABOUT], torch.Tensor([[current_node[1]],[selected_node[1]]]).to(torch.int64).to(device)), dim=1)
                    
            elif current_node[0] == AUTHOR:
                expanded_edges[WRITES] = torch.cat((expanded_edges[WRITES], torch.Tensor([[current_node[1]],[selected_node[1]]]).to(torch.int64).to(device)), dim=1)
                                
            elif current_node[0] == TOPIC:
                expanded_edges[ABOUT] = torch.cat((expanded_edges[ABOUT], torch.Tensor([[selected_node[1]],[current_node[1]]]).to(torch.int64).to(device)), dim=1)

        return selected_node
    else:
        return None
 

def infosphere_noisy_expansion(full_graph, seeds_edges, p1, p2, p3, f, num_seeds, author_node, device):
    # if not seeds_graph[0].nelement() and not seeds_graph[1].nelement() and not seeds_graph[2].nelement(): 
    #     #raise Exception("Not possible to expand an empty infosphere.")
    #     return None

    #print(author_node)
    expanded_edges = [
            torch.tensor([[],[]]).to(torch.int64).to(device),
            torch.tensor([[],[]]).to(torch.int64).to(device),
            torch.tensor([[],[]]).to(torch.int64).to(device)]
    
    node_list = [[], [], []]
    
    p = [0, p1, p2]
    node_to_add = num_seeds * f
    
	
    set_color(seeds_edges, ORANGE, color_tracker, node_list)  # Color all seeds in the infosphere.

    last_node = (4, None)
    current_node = (AUTHOR, author_node)
    current_color = ORANGE
    
    count_path_len = 0

    while node_to_add:
        if random.random() > p[current_color]:
            # p = True, follow the current_color path.
            new_node = expand_seeds(current_node, [current_color], full_graph, color_tracker, exploration_tracker, seeds_edges, expanded_edges, node_list, last_node, device)
            if new_node:
                last_node = current_node
                current_node = new_node
                count_path_len += 1
            else:
                # No current_color nodes available, change direction.
                if current_color == ORANGE:
                    color = [GREEN, WHITE]
                else:
                    color = [WHITE]
                new_node = expand_seeds(current_node, color, full_graph, color_tracker, exploration_tracker, seeds_edges, expanded_edges, node_list, last_node, device)
                if new_node:
                    node_to_add -= 1
                    last_node = current_node
                    current_node = new_node
                    current_color = GREEN
                    count_path_len = 0
                else:
                    node_to_add -= 1
                    count_path_len = 7
        else:
            # p = False, change direction.
            if current_color == ORANGE:
                color = [GREEN, WHITE]
            else:
                color = [WHITE]
            new_node = expand_seeds(current_node, color, full_graph, color_tracker, exploration_tracker, seeds_edges, expanded_edges, node_list, last_node, device)
            if new_node:
                node_to_add -= 1
                last_node = current_node
                current_node = new_node
                current_color = GREEN
                count_path_len = 0
            else:
                # No white nodes available, , follow the current_color path.
                new_node = expand_seeds(current_node, [current_color], full_graph, color_tracker, exploration_tracker, seeds_edges, expanded_edges, node_list, last_node, device)
                if new_node:
                    last_node = current_node
                    current_node = new_node
                    count_path_len += 1
                else:
                    node_to_add -= 1
                    count_path_len = 7
                    
        if random.random() > p3 or count_path_len > 7:
            current_node = (AUTHOR, author_node)
            current_color = ORANGE
            count_path_len = 0
            last_node = (4, None)
    
    for i, node_type in enumerate(node_list):
        for element in node_type:
            exploration_tracker[i][element] = 0
            color_tracker[i][element] = 0
            
    return expanded_edges