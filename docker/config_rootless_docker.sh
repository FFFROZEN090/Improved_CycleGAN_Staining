user = $(whoami)

curl -sSL https://get.docker.com/rootless | sh
export PATH=/home/$user/bin:$PATH
export PATH=$PATH:/sbin
export DOCKER_HOST=unix:///run/user/1027/docker.sock
systemctl --user start docker