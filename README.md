
CycleGAN for Cellular Image Colorization: Technical Documentation
This document provides detailed instructions for setting up and executing the CycleGAN-based cellular image colorization project within a Docker environment. It outlines the creation of a Docker image configured with the necessary dependencies and environment for running the CycleGAN model, as well as steps to deploy a Jupyter Notebook server for interactive development and visualization.

Docker Environment Setup
Dockerfile Configuration
The Dockerfile is designed to establish a foundational environment leveraging NVIDIA's PyTorch Docker image. This setup ensures compatibility with CUDA-enabled GPUs for accelerated model training.

Dockerfile
Copy code
# Base image with NVIDIA PyTorch support
```
FROM nvcr.io/nvidia/pytorch:21.02-py3
```



# Install additional Python packages
```bash
RUN pip install visdom dominate
```



# Install project-specific Python requirements
```bash
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
```



# JupyterLab installation and configuration
```bash
RUN pip install jupyterlab
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
```



# Expose the default Jupyter Notebook port

Requirements File
The requirements.txt file specifies the necessary Python libraries and their versions to ensure compatibility and functionality of the CycleGAN model and the data processing scripts.

shell
Copy code

```bash
--find-links https://download.pytorch.org/whl/torch_stable.html
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
networkx>=2.2
numpy~=1.18.2
pillow~=7.1.1
matplotlib~=3.2.1
opencv-python~=4.2.0.34
requests~=2.23.0
scipy~=1.4.1
scikit-image~=0.19.0
jupyterlab
```

Docker Image Building
The Docker image is built using the provided Dockerfile, tagging it with the current date to facilitate version control and ensure reproducibility.



```bash
docker build -t cv_final:$(date +%Y-%m-%d) .
Docker Configuration for Rootless Operation
To enable rootless operation of Docker, the following configuration script sets up the environment for the current user, allowing Docker commands to be run without root privileges.
```

```bash


user=$(whoami)

curl -sSL https://get.docker.com/rootless | sh
export PATH=/home/$user/bin:$PATH
export PATH=$PATH:/sbin
export DOCKER_HOST=unix:///run/user/1027/docker.sock
systemctl --user start docker
```

Running Jupyter Notebook Server in Docker
The final step involves deploying a Docker container configured to run a Jupyter Notebook server. This server provides an interactive environment for developing and testing the CycleGAN model. The command below mounts the project directory within the container, exposes the Jupyter Notebook server port, and sets various parameters to optimize performance.




```bash
user=$(whoami)
image_name=cv_final
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user
workdir=/home/$user/CV_FinalProject/Cell_cycleGAN

docker run --gpus all --rm -it -w $workdir -v $dir:$dir -p 8888:8888 --shm-size=10g --ulimit memlock=1 --ulimit stack=67108864 $image_id \
jupyter notebook --notebook-dir=$workdir --ip='0.0.0.0' --port=8888 --no-browser --allow-root
```

This command initializes a Jupyter Notebook server within the Docker container, enabling users to access and interact with the project notebooks through a web browser. The configuration facilitates the use of GPU resources for model training, ensuring efficient execution of deep learning tasks.
