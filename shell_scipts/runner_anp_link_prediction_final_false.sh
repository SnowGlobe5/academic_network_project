#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 2 false 50 max 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 3 false 50 max 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 4 false 50 max 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 5 false 50 max 0 0

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 1 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 2 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 3 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 4 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 5 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 2 10 false 50 0 1
