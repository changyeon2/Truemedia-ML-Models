
FROM ubuntu:22.04

ARG SERVING_PORT=8000
ENV SERVING_PORT=$SERVING_PORT

WORKDIR /

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the specified versions of torch and torchvision.
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file and install the requirements.
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
ADD classify.py /classify.py
ADD server.py /server.py
ADD secret.json /secrets/secret.json

# Expose the serving port.
EXPOSE $SERVING_PORT

# Run the server to handle inference requests.
CMD ["python3", "-u", "server.py"]
