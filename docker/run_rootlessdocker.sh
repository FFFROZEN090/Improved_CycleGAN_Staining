user=$(whoami)
image_name=cell_cycle_gan_docker
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user
workdir=/home/$user/Experiments_Repitition/Cell_cycleGAN

docker run --gpus all --rm -it -w $workdir -v $dir:$dir  --shm-size=10g --ulimit memlock=1 --ulimit stack=67108864 $image_id