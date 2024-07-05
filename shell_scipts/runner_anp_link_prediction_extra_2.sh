#!/bin/bash
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_80_mean.py 0.00001 2 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_hgt_80.py 0.00001 2 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_30_mean.py 0.00001 0 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_hgt_30.py 0.00001 0 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_30_sum.py 0.00001 0 50 true