#!/bin/bash
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 0 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 0 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 1 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 1 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 2 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 2 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 4 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 4 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 1 5 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 1 5 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 10 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 2 10 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 2 50 true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 2 50 true

python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [2,5] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [2,5] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,10] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [1,10] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,2] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [5,2] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,1] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [10,1] true
#
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [1,50] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [1,50] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [50,1] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [50,1] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [5,10] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [5,10] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.00001 3 [10,5] true
python3 coauthor_prediction/anp_link_prediction_co_author.py 0.000005 3 [10,5] true




