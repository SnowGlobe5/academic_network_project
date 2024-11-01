#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 1 5 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 2 10 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 3 [5,1] false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 2 50 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 3 [5,10] false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_faster.py 0.0001 0 5 false -1 0 0
