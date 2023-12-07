#!/bin/bash

DATAROOT=/home/frozen/Experiments_Repitition/Cell_cycleGAN/AA/valA
DIRECTION=AtoB
MODEL=test
MODEL_SUFFIX=_B
NUM_TEST=64

for i in {5..60..5}; do
  NAME="/home/frozen/Experiments_Repitition/Cell_cycleGAN/staining_results/training_dataset_tiledGOWT_Fakulty_Inverse_valA/Epoch${i}"
  echo "Running test with NAME=$NAME"
  python ../test.py --dataroot $DATAROOT --name $NAME --model $MODEL --direction $DIRECTION --model_suffix $MODEL_SUFFIX --num_test $NUM_TEST --no_dropout
done
