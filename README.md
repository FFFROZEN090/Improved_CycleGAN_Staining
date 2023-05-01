Before you can run the Code, you need a docker Container.

#Install Docker:

curl -sSL https://get.docker.com/rootless | sh    // run the script on the url to install docker

Then you get following Information:

[INFO] Make sure the following environment variable(s) are set (or add them to ~/.bashrc):

export PATH=/home/rhack/bin:$PATH

[INFO] Some applications may require the following environment variable too:

export DOCKER_HOST=unix:///run/user/1027/docker.sock

run both export commands!

Then you can start docker with:

systemctl --user start docker

#The next step is to build the docker image:

create a folder, name it "docker" and create a file withe the name "Dockerfile"

copy the requirements.txt file into the "docker" folder

copy this into the Dockerfile:

FROM nvcr.io/nvidia/pytorch:21.02-py3


RUN set -ex

RUN pip install visdom

RUN pip install dominate

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

//the two last commands are necessary to use the requirements.txt file

#The last step is to run the docker image:

docker run --gpus all --rm -it --name CycleGAN -w /home/rhack/stainTransfer_CycleGAN_pytorch -v /home/rhack:/home/rhack image_ID

//you get the image id with the command "docker images", replace image_ID with the image id you get

