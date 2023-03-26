#!/bin/bash

python -m molbart.evaluate \
  --data_path ./data/uspto_50.pickle \
  --model_path tb_logs/backward_prediction/version_270_Chemformer/checkpoints/last.ckpt \
  --dataset uspto_50 \
  --task backward_prediction \
  --batch_size 64 \
  --num_beams 10 \
  --model_type bart \
  # --reaction_model_path tb_logs/backward_prediction/version_126_predict_reaction_type_org_10class/checkpoints/last.ckpt \
  # --model_type bartAddReactionType\
  # --model_type bart \
  

  # --reaction_model_path tb_logs/backward_prediction/version_126_predict_reaction_type_org_10class/checkpoints/last.ckpt \
  # tb_logs/backward_prediction/version_147/checkpoints/last.ckpt \
  # --model_type reactionType \

