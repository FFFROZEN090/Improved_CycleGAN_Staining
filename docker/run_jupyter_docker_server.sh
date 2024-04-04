user=$(whoami)
image_name=cv_final
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user
workdir=/home/$user/CV_FinalProject/Cell_cycleGAN

# Run Docker container with Jupyter Notebook
docker run --gpus all --rm -it -w $workdir -v $dir:$dir -p 8888:8888 --shm-size=10g --ulimit memlock=1 --ulimit stack=67108864 $image_id \
jupyter notebook --notebook-dir=$workdir --ip='0.0.0.0' --port=8888 --no-browser --allow-root