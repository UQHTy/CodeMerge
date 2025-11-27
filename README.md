# CodeMerge: Codebook-Guided Model Merging for Robust Test-Time Adaptation in Autonomous Driving

Official code release for the paper "**CodeMerge: Codebook-Guided Model Merging for Robust Test-Time Adaptation in Autonomous Driving**". 


<img src="docs/vis.png" width="600" />


## Abstract
Maintaining robust 3D perception under dynamic and unpredictable test-time conditions remains a critical challenge for autonomous driving systems. Existing test-time adaptation (TTA) methods often fail in high-variance tasks like 3D object detection due to unstable optimization and sharp minima. While recent model merging strategies based on linear mode connectivity (LMC) offer improved stability by interpolating between fine-tuned checkpoints, they are computationally expensive, requiring repeated checkpoint access and multiple forward passes. In this paper, we introduce CodeMerge, a lightweight and scalable model merging framework that bypasses these limitations by operating in a compact latent space. Instead of loading full models, CodeMerge represents each checkpoint with a low-dimensional fingerprint derived from the source model’s penultimate features and constructs a key-value codebook. We compute merging coefficients using ridge leverage scores on these fingerprints, enabling efficient model composition without compromising adaptation quality. Our method achieves strong performance across challenging benchmarks, improving end-to-end 3D detection 14.9\% NDS on nuScenes-C and LiDAR-based detection by over 7.6\% mAP on nuScenes-to-KITTI, while benefiting downstream tasks such as online mapping, motion prediction and planning even without training. Code and pretrained models are released in the supplementary material.

<img src="docs/codemerge.png" width="500" />

## Installation

### Environment

All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04.6 LTS)
* Python 3.8
* PyTorch 1.13.0
* CUDA 11.6
* mmcv-full==1.4.0
* mmdet==2.14.0
* mmsegmentation==0.14.1
* mmdet3d==v0.17.1

All packages/libraries tested for this project have been exported to in [requirements.txt](requirements.txt).


### Quick Start

```shell scripts
sh scripts/test.sh
```
