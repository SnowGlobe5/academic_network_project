#!/bin/bash

#python3 anp_link_prediction_co_author_infosphere_no_exp.py True
python3 anp_link_prediction_co_author_no_infosphere.py False 0.00001
python3 anp_link_prediction_co_author_no_infosphere.py False 0.000001

python3 anp_link_prediction_co_author_infosphere.py False 0.00001 1
python3 anp_link_prediction_co_author_infosphere.py False 0.000001 1
