DATAROOT=/home/frozen/Experiments_Repitition/Cell_cycleGAN/training_dataset_GOWT_Inverse
NAME=/home/frozen/Experiments_Repitition/Cell_cycleGAN/staining_results/GOWT_Inverse/Epoch60
DIRECTION=AtoB
MODEL=test
MODEL_SUFFIX=_B

python test.py --dataroot $DATAROOT --name $NAME --model $MODEL --direction $DIRECTION --model_suffix $MODEL_SUFFIX --no_dropout