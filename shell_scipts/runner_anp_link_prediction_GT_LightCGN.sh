#!/bin/bash
# cd ../anp_core
# python3 get_author_from_recbole.py 1 1
cd ../anp_nn
python3 coauthor_prediction/create_gt_opt_TOP_TOPIC.py
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 4 1 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 5 1 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 2 10 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 2 50 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 3 [5,10] true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 3 [5,2] true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 1 5 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_N.py 0.0001 0 5 true 100 0 0
# python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_TOP.py 0.0001 4 1 true 100 0 0
# python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_2.py 0.0001 4 1 true 100 0 0
# python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.0001 4 1 true 100 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_L.py 0.0001 4 1 true 100 0 0
















