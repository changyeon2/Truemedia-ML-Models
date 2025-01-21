# Exploring Temporal Coherence for More General Video Face Forgery Detection(FTCN) 

Yinglin Zheng, Jianmin Bao, Dong Chen, Ming Zeng, Fang Wen

Accepted by ICCV 2021

### [Paper](https://arxiv.org/abs/2108.06693)  

## Abstract
> Although current face manipulation techniques achieve impressive performance regarding quality and controllability, they are struggling to generate temporal coherent face videos. In this work, we explore to take full advantage of the temporal coherence for video face forgery detection. To achieve this, we propose a novel end-to-end framework, which consists of two major stages. The first stage is a fully temporal convolution network (FTCN). The key insight of FTCN is to reduce the spatial convolution kernel size to 1, while maintaining the temporal convolution kernel size unchanged. We surprisingly find this special design can benefit the model for extracting the temporal features as well as improve the generalization capability. The second stage is a Temporal Transformer network, which aims to explore the long-term temporal coherence. The proposed framework is general and flexible, which can be directly trained from scratch without any pre-training models or external datasets. Extensive experiments show that our framework outperforms existing methods and remains effective when applied to detect new sorts of face forgery videos.


# Setup
First setup python environment with pytorch 1.4.0 installed, **it's highly recommended to use docker image [pytorch/pytorch:1.4-cuda10.1-cudnn7-devel](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-devel/images/sha256-c612782acc39256aac0637d58d297644066c62f6f84f0b88cfdc335bb25d0d22), as the pretrained model and the code might be incompatible with higher version pytorch.**

then install dependencies for the experiment:

```
pip install -r requirements.txt
```

# Test

## Inference Using Pretrained Model on Raw Video
Download `FTCN+TT` model trained on FF++ from [here](https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth) and place it under `./checkpoints` folder
```bash
python test_on_raw_video.py examples/shining.mp4 output
```
the output will be a video under folder `output` named `shining.avi`

![](./examples/shining.gif)

# TODO

- [x] Release inference code.
- [ ] Release training code.
- [ ] Code cleaning.


# Acknowledgments

This code borrows heavily from [SlowFast](https://github.com/facebookresearch/SlowFast).

The face detection network comes from [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface).

The face alignment network comes from [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark).



# Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{zheng2021exploring,
  title={Exploring Temporal Coherence for More General Video Face Forgery Detection},
  author={Zheng, Yinglin and Bao, Jianmin and Chen, Dong and Zeng, Ming and Wen, Fang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15044--15054},
  year={2021}
}
```

# TrueMedia
## Training
### Preprocess
We have to preprocess the training data first.
```
python training_preprocessing.py
  -d <dir-path>
  -o <output-dir>
  -c <config>
  --max_clips <max_clips>
  --max_frames <max_frames>
  --face_thres <face_thres>
  --bbox-thres <bbox_thres>
```
`<data-path>` : Path to the data directory. <br/>
`<output-dir>`: Path to the output directory. <br/>
`<config>`: Path to the configuration file (i.e. ftcn_tt.yaml).<br/>
`<max_clips>`: The max number of clips to take from the video. <br/>
`<max_frames>`: The max number of frames to take from the video. <br/>
`<face_thres>`: The threshold for face detector. <br/>
`<bbox_thres>`: The bounding box threshold for face detector. <br/>

For each `fake` and `real` directories in your raw dataset run the script.

```
python training_preprocessing.py \ 
  -d <dir-path>/real \
  -o <output-dir>/real

python training_preprocessing.py \
  -d <dir-path>/fake \
  -o <output-dir>/fake
```

### Train
```
input:
<raw-dataset>
  - fake
  - real

output:
<preprocessed-dataset>
  - fake
  - real
```

```
python train.py 
  -d <training-data-path>
  -p <model-pretrained>
  -c <config>
  --split <train-val-split>
  --freeze
```

`<training-data-path>` : Path to the preprocessed training data. Directory has to be structured as such:
```
<preprocessed-dataset>
  - fake
  - real
```
`<model-pretrained>`: Path to pretrained/checkpoint weights you want to start training with (i.e. checkpoints/).<br/>
`<config>`: Path to the configuration file (i.e. ftcn_tt.yaml).<br/>
`<train-val-split>`: Path to the train and val split. (optional)
`--freeze`: Freeze backbone. (optional)

```
python train.py \
  -d s3://ftcn-dataset/lin-finetune \
  -p checkpoints/ftcn_tt.pth \
  --split /home/ubuntu/iman/global-ml/FTCN/dataset_splits.csv \
  --freeze
```
## Deploy

### Test locally.
```
python -u -m server
curl -X GET http://localhost:8000/healthcheck
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    --data '{"file_path":"https://www.evalai.org/ocasio.mp4"}'
```

### with Docker 

```
export DOCKER_REGISTRY="miraflow" # Put your Docker Hub username here  
docker build -t "$DOCKER_REGISTRY/ftcn-finetune" -f Dockerfile .
```

Push your docker image to docker hub
```
docker login

docker push "$DOCKER_REGISTRY/ftcn-finetune"
```


Run this Docker image locally on a GPU to test that it can run inferences as expected:
```
docker run --gpus=all -d --rm -p 80:8000 --env SERVER_PORT=8000  --name "ftcn" "$DOCKER_REGISTRY/ftcn-finetune"
```

map port 80 on your host to port 8000 in the Docker container. AWS only allows 80.

..and in a separate terminal run the following command one or more times

```
curl -X GET http://localhost:80/healthcheck
```
until you see {"healthy":true}

```
curl -X POST http://localhost:80/predict \
    -H "Content-Type: application/json" \
    --data '{"file_path":"https://www.evalai.org/ocasio.mp4"}'
```
If running on server you can also use your server url instead of localhost:
```
curl -X POST http://<SERVER IP>:80/predict \
    -H "Content-Type: application/json" \
    --data '{"file_path":"https://www.evalai.org/ocasio.mp4"}'
```