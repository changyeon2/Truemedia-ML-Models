# 참고링크 : https://jishuzhan.net/article/1780499919056932866 
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# check
ARG SERVING_PORT=8002
ENV SERVING_PORT=$SERVING_PORT


WORKDIR /

###################################

# Copy in the files necessary to fetch, run and serve the model.
ADD checkpoints/content/true_best_1118.pth /checkpoints/content/true_best_1118.pth
ADD data/ /data/
# ADD dataset/ /dataset/


# https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub



RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt search Xlib.h


# Upgrade pip and install the copied in requirements.
RUN pip install --no-cache-dir --upgrade pip

ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

###################################

# add the least frequently modified files first and the most frequently modified files last. 

ADD test_tools/ /test_tools/
ADD preprocessing_utils.py /preprocessing_utils.py

ADD model/ /model/
ADD slowfast/ /slowfast/
ADD setting/ /setting/
ADD root_setting.yaml /root_setting.yaml 
ADD config_ftcn.py /config_ftcn.py
ADD custommodel.py /custommodel.py
ADD server.py /server.py

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Fetch the model and cache it locally.
RUN python -m custommodel --fetch -p https://www.evalai.org/ocasio.mp4

# Expose the serving port.
EXPOSE $SERVING_PORT

# Add / to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/"

# Run the server to handle inference requests.
CMD python3 -u -m server
