#!/bin/bash

python -m molbart.evaluate \
  --data_path ./data/uspto_50.pickle \
  --model_path tb_logs/backward_prediction/version_121_predict_reaction_type_prompt_10class_use_reaClass/checkpoints/last.ckpt \
  --dataset uspto_50 \
  --task backward_prediction \
  --model_type reactionType \
  --batch_size 64 \
  --num_beams 10

