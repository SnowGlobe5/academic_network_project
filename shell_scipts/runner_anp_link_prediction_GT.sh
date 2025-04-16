#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_TT.py 0.00001 1 5 true 50 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_TT.py 0.00001 0 5 true 50 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 2 50 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT.py 0.00001 3 [5,10] true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_TT.py 0.00001 2 50 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_GT_TT.py 0.00001 3 [5,10] true -1 0 0










