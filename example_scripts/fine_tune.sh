

python -m molbart.fine_tune \
  --dataset uspto_50 \
  --data_path data/uspto_50.pickle \
  --model_path models/combined/step=1000000.ckpt \
  --task backward_prediction \
  --epochs 300 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5

