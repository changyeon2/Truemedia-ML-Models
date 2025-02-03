[Update February 3, 2025] We haven’t shared the model weights with anyone yet, as we’re still carefully working on a responsible release policy to ensure security and prevent any potential misuse. Thank you for your understanding.

> This code is published as-is for reference and educational purposes in the field of deepfake detection. It represents a historical implementation by TrueMedia.org and is not actively maintained. The repository does not accept pull requests or support requests. 

# TrueMedia.org ML Models Collection

This repository contains a collection of advanced machine learning models and deepfake detectors. The models were developed by [TrueMedia.org](https://www.truemedia.org/), a non-profit organization committed to detecting political deepfakes and supporting fact-checking efforts for the 2024 election.

Each detector is contained in its own directory with specific documentation and implementation details.

## Models

### 1. [DistilDIRE](/DistilDIRE) (image)

Distil-DIRE is a lightweight version of DIRE, which can be used for real-time applications. Instead of calculating DIRE image directly, Distl-DIRE aims to reconstruct the features of corresponding DIRE image forwared by a image-net pretrained classifier with one-step noise of DDIM inversion. ([Paper Link](https://arxiv.org/abs/2406.00856))

- Utilizes a knowledge distillation approach from pre-trained diffusion models.
- Achieves a 3.2x faster inference speed compared to the traditional DIRE framework.
- Capable of detecting diffusion-generated images as well as those produced by GANs
 
### 2. [UniversalFakeDetectV2](/UniversalFakeDetectV2) (image)

An updated implementation of the "UniversalFakeDetect" model by [Ojha et al.](https://arxiv.org/abs/2302.10174) for detecting AI-generated images. Key features:

- Built on HuggingFace CLIP models
- Simplified codebase to allow training on large image datasets (100k+)
- Streamlined training and testing scripts
- Docker deployment support

### 3. [GenConViT](https://github.com/truemediaorg/GenConViT) (video)

An updated implementation of the GenConViT model by [Wodajo et al.](https://arxiv.org/abs/2307.07036). Features:

- Support for both VAE and encoder-decoder architectures
- FP16 precision support
- Frame-based video processing capabilities
- CLI interface for easy model interaction

### 4. [StyleFlow](/StyleFlow) (video)

A model for temporal-aware style transfer and manipulation ([Paper Link](https://openaccess.thecvf.com/content/CVPR2024/papers/Choi_Exploiting_Style_Latent_Flows_for_Generalizing_Deepfake_Video_Detection_CVPR_2024_paper.pdf)). Includes:

- Time transformer architecture
- Face-aware processing
- Video frame processing capabilities
- ResUNet implementation for feature extraction

### 5. [Reverse Research](/reverse-search) (image)

A reverse image search pipeline for deepfake detection that leverages multiple APIs:

- Google Vision API integration
- Support for Perplexity and GPT-4
- Docker containerization
- Health monitoring endpoints

### 6. [FTCN](/FTCN) (video)

An updated implementation of the "FTCN" model used for detecting deepfakes ([Paper Link](https://arxiv.org/abs/2108.06693)):

- Aims to explore the long-term temporal coherence with temporal transformer network
- Fully temporal convolution network
- Extracts temporal features
- Anaylzes facials features

### 7. [Transcript Based Detector](/transcript) (audio)

- Uses a speach-recognition model and a LLM to make predictions on audio veracity
- Predictions are based on language construction, narrative coherence, and factual alignment

## Setup

Each model has its own setup requirements and dependencies. Please refer to the individual directories for specific setup instructions.

## Accessing Model Weights for Research

We believe in advancing responsible AI development through collaboration. To receive the trained weights for our deepfake detection models:

1. Email aerin@truemedia.org with subject "Weight Access Request"

2. Include:

   - Your affiliation
   - Intended usage purpose
   - Research/project outline
   - Agreement to ethical guidelines

Access is granted to verified users only. We carefully review each request to ensure responsible usage of these tools.

## Licenses

This project is licensed under the terms of the MIT license. Note that our [GenConViT model is licensed differently](https://github.com/truemediaorg/GenConViT) in a separate repository.

## Contributors

The Machine Learning engineers at TrueMedia.org improved open-source models and also developed new detectors. We're now making these advancements available to the community.

[Aerin Kim](https://github.com/aerinkim)

[Arnab Karmakar](https://github.com/arnabkuw)

[Ben Caffee](https://github.com/bcaffee)

[Hannah Lee](https://github.com/hannahyklee)

[Iman Tanumihardja](https://github.com/ImanTanumihardja)

[Jongwook Choi](https://github.com/jongwook-Choi)

[Kevin Farhat](https://github.com/kevin-farhat)

[Lin Qiu](https://github.com/linqiu0-0)

[Max Bennett](https://github.com/maxmiles)

[Nuria Alina Chandra](https://github.com/nuriachandra)

[Yewon Lim](https://github.com/yevvonlim)

[Changyeon Lee](https://github.com/changyeon2)
