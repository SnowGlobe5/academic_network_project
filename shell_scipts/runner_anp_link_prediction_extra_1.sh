#!/bin/bash
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_80_max.py 0.00001 2 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_80_min.py 0.00001 2 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_30_max.py 0.00001 0 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_30_min.py 0.00001 0 50 true
python3 coauthor_prediction/extra_edge_issue/anp_link_prediction_co_author_80_sum.py 0.00001 2 50 true