FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

ARG SERVING_PORT=8000
ENV SERVING_PORT=$SERVING_PORT

WORKDIR /

ENV DEBIAN_FRONTEND noninteractive

# Update package list, install the OpenGL library, libglib2.0-0, and clear the APT lists
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the copied in requirements.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# add the least frequently modified files first and the most frequently modified files last.
ADD utils/ /utils/
ADD config.py /config.py
ADD test_tools/ /test_tools/
ADD slowfast/ /slowfast/
ADD root_setting.yaml /root_setting.yaml
ADD setting/ftcn_tt.yaml /setting/ftcn_tt.yaml
ADD checkpoints/ /checkpoints/
ADD model/ /model/
ADD custommodel.py /custommodel.py
ADD server.py /server.py

# Fetch the model and cache it locally.
# RUN python3 custommodel.py --fetch -p https://www.evalai.org/ocasio.mp4

# Expose the serving port.
EXPOSE $SERVING_PORT

# Add / to the PYTHONPATH
ENV PYTHONPATH="/"

# Run the server to handle inference requests.
CMD python3 -u server.py
