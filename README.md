# TrueMedia ML Models Collection

This repository contains a collection of advanced machine learning models for various computer vision and generative AI tasks. Each model is contained in its own directory with specific documentation and implementation details.

## Models

### 1. DistilDIRE
A diffusion model implementation based on the paper "Diffusion Models Beat GANs on Image Synthesis". Features include:
- Class-conditional and unconditional image generation
- Multiple resolution support (64x64 to 512x512)
- Classifier guidance capabilities
- Support for both ImageNet and LSUN datasets

### 2. UniversalFakeDetectV2
An updated implementation of the "UniversalFakeDetect" model for detecting AI-generated images. Key features:
- Built on HuggingFace CLIP models
- Simplified codebase for specific detection goals
- Streamlined training and testing scripts
- Docker deployment support

### 3. GenConViT
A generative model with Vision Transformer architecture. Features:
- Support for both VAE and encoder-decoder architectures
- FP16 precision support
- Frame-based video processing capabilities
- CLI interface for easy model interaction

### 4. StyleFlow
A model for temporal-aware style transfer and manipulation. Includes:
- Time transformer architecture
- Face-aware processing
- Video frame processing capabilities
- ResUNet implementation for feature extraction

### 5. Reverse Research

A reverse image search pipeline for deepfake detection that leverages multiple APIs:
- Google Vision API integration
- Support for Perplexity and GPT-4
- Docker containerization
- Health monitoring endpoints


## Setup

Each model has its own setup requirements and dependencies. Please refer to the individual model directories for specific setup instructions.

## License

This project is licensed under the terms included in the LICENSE file.

## Contributors

The ML engineers at TrueMedia worked on open-source models and made their original models better. We're sharing our improvement with the community.

[Aerin Kim](https://github.com/aerinkim) 

[Arnab Karmakar](https://github.com/arnabkuw) 

[Ben Caffee](https://github.com/bcaffee) 

[Iman Tanumihardja](https://github.com/ImanTanumihardja) 

[Jongwook Choi](https://github.com/jongwook-Choi) 

[Kevin Farhat](https://github.com/kevin-farhat) 

[Lin Qiu](https://github.com/linqiu0-0) 

[Nuria Alina Chandra](https://github.com/nuriachandra) 

[Hannah Lee](https://github.com/hannahyklee) 

[Max Miles](https://github.com/maxmiles) 
