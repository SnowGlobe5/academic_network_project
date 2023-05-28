# once the paths have been computed  you can  create the actual infosphere by using an extension policy. the base  policy would start from the pahts and choose randomly
# a node and randomly expand a few children. No need to check for duplicates. the only parameters would be how many nodes to expand and the maximum children node to expand.

# please keep node selection and children expansions part of the policy separated as they can be later specialised by learning or by taking into account domain specific
# heuristics like open the most cited papers etc

def random_selection_policy():
    pass

def random_expansion_policy():
    pass

def expand_infosphere(n_node, n_children, selection_policy, expansion_policy):
    pass