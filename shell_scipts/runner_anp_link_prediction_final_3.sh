#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 true 50 sum 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 false 50 sum 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 true 50 min 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 false 50 min 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 true 50 max 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 false 50 max 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 true 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 false 50 mean 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 true -1 sum 0 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 0 0 false -1 sum 0 0


