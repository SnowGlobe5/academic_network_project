#!/bin/bash

python3 noisy-caller.py 1 > noisy1.log
python3 anp_build_infosphere.py 1

python3 noisy-caller.py 2 > noisy2.log
python3 anp_build_infosphere.py 2

python3 noisy-caller.py 3 > noisy3.log
python3 anp_build_infosphere.py 3

python3 noisy-caller.py 4 > noisy4.log
python3 anp_build_infosphere.py 4

python3 noisy-caller.py 5 > noisy5.log
python3 anp_build_infosphere.py 5


python3 anp_link_prediction_co_author_infosphere.py False 1
python3 anp_link_prediction_co_author_infosphere.py False 2
python3 anp_link_prediction_co_author_infosphere.py False 3
python3 anp_link_prediction_co_author_infosphere.py False 4
python3 anp_link_prediction_co_author_infosphere.py False 5
