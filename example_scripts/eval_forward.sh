#!/bin/bash

python -m molbart.evaluate \
  --data_path ./data/USPTO_full/uspto_full.pickle \
  --model_path tb_logs/backward_prediction/version_259_avg_type_loss_by_batch_weight_1_decoder/checkpoints/last.ckpt \
  --dataset uspto_50 \
  --task backward_prediction \
  --batch_size 128 \
  --num_beams 10 \
  --model_type bart \
  # --reaction_model_path tb_logs/backward_prediction/version_126_predict_reaction_type_org_10class/checkpoints/last.ckpt \
  # --model_type bartAddReactionType\
  # --model_type bart \
  

  # --reaction_model_path tb_logs/backward_prediction/version_126_predict_reaction_type_org_10class/checkpoints/last.ckpt \
  # tb_logs/backward_prediction/version_147/checkpoints/last.ckpt \
  # --model_type reactionType \

