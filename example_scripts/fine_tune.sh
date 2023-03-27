python -m molbart.fine_tune \
  --dataset uspto_50 \
  --data_path data/uspto_50.pickle \
  --model_path models/combined/step=1000000.ckpt \
  --task backward_prediction \
  --epochs 100 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 \
  --reaction_model_path tb_logs/backward_prediction/version_259_avg_type_loss_by_batch_weight_1_decoder/checkpoints/last.ckpt
  # --model_type reactionType\
  
  
  # --model_path models/combined/step=1000000.ckpt \
  # --model_path None \
  # --model_type reactionType\
  # --reaction_model_path tb_logs/backward_prediction/version_126_predict_reaction_type_org_10class/checkpoints/last.ckpt
  # --model_type reactionType\
  
  # --d_model 1024 \
  # --num_layers 8 \
  # --d_feedforward 4096 \
  # --num_heads 16

