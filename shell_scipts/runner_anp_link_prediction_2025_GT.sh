#!/bin/bash
#python3 coauthor_prediction/create_GT/create_gt_opt.py 2 10 false
#python3 coauthor_prediction/create_GT/create_gt_opt.py 3 5_2 false
#python3 coauthor_prediction/create_GT/create_gt_opt.py 4 0 false

cd /data/sabrina
python3 academic_network_project/anp_core/check_history.py 2 10 
python3 academic_network_project/anp_core/check_history.py 3 5_2
python3 academic_network_project/anp_core/check_history.py 4 0