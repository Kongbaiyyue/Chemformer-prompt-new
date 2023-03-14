python -m molbart.fine_tune \
  --model_type reactionType \
  --dataset uspto_50_with_type \
  --data_path data/uspto_50.pickle \
  --model_path models/combined/step=1000000.ckpt \
  --task backward_prediction \
  --epochs 400 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 
  # --d_model 1024 \
  # --num_layers 8 \
  # --d_feedforward 4096 \
  # --num_heads 16

