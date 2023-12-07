DATAROOT=/home/frozen/Experiments_Repitition/Cell_cycleGAN/Training_Datasets/Blue_Bubble_tiledGOWT_Inverse
RESULT_DIR=/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/Blue_Bubble_tiledGOWT_Inverse
NAME=cycle_gan
MODEL=cycle_gan
BATCH_SIZE=24
EPOCHS=200
EPOCHS_DECAY=200

python ../train.py --dataroot $DATAROOT --name cycle_gan --results_dir $RESULT_DIR --name $NAME \
                   --model $MODEL --batch_size $BATCH_SIZE --n_epochs $EPOCHS \
                   --n_epochs_decay $EPOCHS_DECAY --continue_train --epoch_count 367