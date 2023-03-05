#!/bin/bash

python setup.py develop
python -m molbart.train \
  --dataset USPTOPretrain \
  --data_path ./data/uspto_50_pretrain \
  --model_type bart \
  --lr 1.0 \
  --schedule transformer \
  --epochs 1000 \
  --batch_size 128 \
  --gpus 1 \
  --task mask_aug

