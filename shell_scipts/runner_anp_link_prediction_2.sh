#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 0 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 1 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 1 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 2 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 2 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 4 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 4 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 5 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 5 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 2 10 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 2 50 false

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [2,5] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [2,5] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [1,10] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,2] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [5,2] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,1] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [10,1] false
#
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [1,50] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [50,1] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [50,1] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,10] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [5,10] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,5] false
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [10,5] false




