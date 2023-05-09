set -ex

START=$(date +%s.%N)

python train.py --dataroot /home/rhack/stainTransfer_CycleGAN_pytorch/to/data --name cycle_gan --results_dir /home/rhack/stainTransfer_CycleGAN_pytorch --name cycle_gan --load_size {load_size} --crop_size {crop_size} --pool_size {pool_size} --model cycle_gan

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
