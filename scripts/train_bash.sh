DATAROOT=/home/frozen/Experiments_Repitition/Cell_cycleGAN/training_dataset_GOWT_Inverse
RESULT_DIR=/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/GOWT_Inverse
NAME=cycle_gan
MODEL=cycle_gan
BATCH_SIZE=16

python train.py --dataroot $DATAROOT --name cycle_gan --results_dir $RESULT_DIR --name $NAME  --model $MODEL --batch_size $BATCH_SIZE