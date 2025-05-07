#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 2 50 false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 2 [5,10] false -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 1 5 false -1 0 0

python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 1 5 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 2 50 true -1 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_hgt_embedding_64.py 0.00001 2 [5,10] true -1 0 0