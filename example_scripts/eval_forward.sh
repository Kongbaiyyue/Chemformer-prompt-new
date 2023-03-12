#!/bin/bash

python -m molbart.evaluate \
  --data_path ./data/uspto_50.pickle \
  --model_path tb_logs/backward_prediction/version_1_use_all_type_token/checkpoints/last.ckpt \
  --dataset uspto_50_with_type \
  --task backward_prediction \
  --model_type bart \
  --batch_size 64 \
  --num_beams 10

