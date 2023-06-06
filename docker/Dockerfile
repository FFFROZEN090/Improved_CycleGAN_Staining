 
#FROM nvcr.io/nvidia/pytorch:23.04-py3
#FROM nvcr.io/nvidia/pytorch:20.11-py3
FROM nvcr.io/nvidia/pytorch:21.02-py3


RUN set -ex
# install pytorch (1.7.1) for CUDA 11.0
#pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# install pytorch (1.7.1) for CUDA 10.2
#pip install torch torchvision
# install visdom
RUN pip install visdom
# install dominate
RUN pip install dominate
#RUN find . -regex '.*requirements.txt$'
#RUN pip freeze > requirements.txt
#RUN pip install -r requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt
