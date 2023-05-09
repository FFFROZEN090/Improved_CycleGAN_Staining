# Git commands:
Git global setup:

git config --global user.name "rhack"

git config --global user.email "robin.hack@stud.uni-heidelberg.de"

Existing Folder:

cd existing folder
git init

git remote add origin http://gitlab.medma.uni-heidelberg.de/mbai_dmri/Cell_cycleGAN.git

git add .

git commit -m "some comment"

git push -u origin master

If hard reset is necessary:

git reset --hard origin/<branch_name>

# Prüfungs und Studienordnung Informatik

https://backend.uni-heidelberg.de/de/dokumente/studien-und-pruefungsordnung-informatik-ba-2022-10-05/download

# Frequently asked questions and tipps and the paper and Pytorch images:

Questions: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

Tipps: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md

The Paper: https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-021-01126-y#Sec25

Nvidia Pytorch images: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

# Before you can run the Code, you need a docker Container.

# Install Docker:

curl -sSL https://get.docker.com/rootless | sh    // run the script on the url to install docker

Then you get following Information:

_____________________________________________________________________________________________

[INFO] Make sure the following environment variable(s) are set (or add them to ~/.bashrc):

export PATH=/home/rhack/bin:$PATH

[INFO] Some applications may require the following environment variable too:

export DOCKER_HOST=unix:///run/user/1027/docker.sock

___________________________________________________________________________________________________

copy both export commands into the shell and run them!

Then you can start docker with:

systemctl --user start docker

# The next step is to build the docker image:

create a folder, name it "docker" and create a file in it with the name "Dockerfile"

copy the requirements.txt file into the "docker" folder

copy this into the Dockerfile:

_________________________________________

FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN set -ex

RUN pip install visdom

RUN pip install dominate

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt
_______________________________________________________

//the two last commands are necessary to use the requirements.txt file

navigate into the docker folder and run: docker build .

# The last step is to run the docker image:

docker run --gpus all --rm -it --name CycleGAN -w /home/rhack/stainTransfer_CycleGAN_pytorch -v /home/rhack:/home/rhack image_ID

//you get the image id with the command "docker images", replace image_ID with the image id you get

# Prepare your own data for CycleGAN

You need to create two directorys to host images from domain A (/path/to/data/trainA) and from domain B (/path/to/data/trainB)

# Than you can train the model with:

python train.py --dataroot {path_to_trainA_and_trainB} --name {name_of_experiment} --results_dir {path_to_results} --name {name_of_experiment} --load_size {load_size} --crop_size {crop_size} --pool_size {pool_size} --model cycle_gan

I have used this actually(04.05.23):

python train.py --dataroot /home/rhack/stainTransfer_CycleGAN_pytorch/to/data --name cycle_gan --results_dir /home/rhack/stainTransfer_CycleGAN_pytorch --name cycle_gan  --model cycle_gan

# Training Options/Test Options

In the folder options" is a file "train_opt.py" were you can change the training options.

In the folder options" is a file "test_opt.py" were you can change the test options.

# Plot of Training Result:

To view training results and loss plots, run: python -m visdom.server and click the URL http://localhost:8097.

# Test the model:

Test a CycleGAN model (both sides):
        python test.py --dataroot /home/rhack/stainTransfer_CycleGAN_pytorch/to/data --name maps_cyclegan --model cycle_gan
        
    Test a CycleGAN model (one side only):
        python test.py --dataroot /home/rhack/stainTransfer_CycleGAN_pytorch/to/data/trainA --name cycle_gan_pretrained --model test --no_dropout
