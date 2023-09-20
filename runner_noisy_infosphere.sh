#!/bin/bash

python3 noisy-caller.py 2 > noisy2.log

while true; do
  if [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_1.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_2.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_1_1_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_2_3_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_4_4_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_5_9_None.pt ]
  then
      break
  fi
  sleep 60
done

python3 anp_build_infosphere.py 2


python3 noisy-caller.py 4 > noisy4.log

while true; do
  if [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_1.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_2.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_1_1_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_2_3_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_4_4_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_5_9_None.pt ]
  then
      break
  fi
  sleep 60
done

python3 anp_build_infosphere.py 4


python3 noisy-caller.py 5 > noisy5.log

while true; do
  if [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_1.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_0_0_2.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_1_1_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_2_3_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_4_4_None.pt ] && [ -e ANP_DATA/computed_infosphere/infosphere_0_1_2_3_4_2019_noisy_5_9_None.pt ]
  then
      break
  fi
  sleep 60
done

python3 anp_build_infosphere.py 5
