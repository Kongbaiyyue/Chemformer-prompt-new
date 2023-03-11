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
  --aug_prob 0.5 
  # --d_model 1024 \
  # --num_layers 8 \
  # --d_feedforward 4096 \
  # --num_heads 16

