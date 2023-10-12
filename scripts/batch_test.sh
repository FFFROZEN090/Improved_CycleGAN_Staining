#!/bin/bash

DATAROOT=/home/frozen/Experiments_Repitition/Cell_cycleGAN/AA/testA
DIRECTION=AtoB
MODEL=test
MODEL_SUFFIX=_B
NUM_TEST=576

for i in {5..60..5}; do
  NAME="/home/frozen/Experiments_Repitition/Cell_cycleGAN/staining_results/GOWT_Inverse/Epoch${i}"
  echo "Running test with NAME=$NAME"
  python ../test.py --dataroot $DATAROOT --name $NAME --model $MODEL --direction $DIRECTION --model_suffix $MODEL_SUFFIX --num_test $NUM_TEST --no_dropout
done
