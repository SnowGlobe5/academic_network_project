#!/bin/bash
python3 get_author_from_top_paper_per_topic.py 1 1
cd ../anp_nn
python3 coauthor_prediction/create_gt_opt_paper.py
