#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false 50 min 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false 50 mean 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false 50 max 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false -1 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 2 50 false 50 0 1

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false 50 min 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false 50 max 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false 50 mean 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false -1 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 3 [1,10] false 50 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,1] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [2,5] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,2] false 50 sum 0 1
#
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false 50 min 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false 50 max 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false 50 mean 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false -1 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author_htg.py 0.00001 3 [1,50] false 50 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [50,1] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,10] false 50 sum 0 1
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,5] false 50 sum 0 1




