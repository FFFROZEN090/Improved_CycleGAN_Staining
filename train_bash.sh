DATAROOT=/home/jli/Cell_cycleGAN/data_training
RESULT_DIR=/home/jli/Cell_cycleGAN/results
NAME=cycle_gan
MODEL=cycle_gan

python train.py --dataroot $DATAROOT --name cycle_gan --results_dir $RESULT_DIR --name $NAME  --model $MODEL