#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 min 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 mean 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 max 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true -1 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 1 0 true 50 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 sum 0.5
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 sum 0.25
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true 50 sum 0.1

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 1 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 2 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 3 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 4 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 5 true 50 sum 0

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true 50 min 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true 50 mean 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true 50 max 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true -1 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 2 10 true 50 0

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true 50 min 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true 50 mean 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true 50 max 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true -1 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 2 50 true 50 0

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true 50 min 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true 50 max 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true 50 mean 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true -1 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 3 [1,10] true 50 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,1] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [2,5] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,2] true 50 sum 0
#
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true 50 min 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true 50 max 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true 50 mean 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true -1 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 3 [1,50] true 50 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [50,1] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,10] true 50 sum 0
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,5] true 50 sum 0




