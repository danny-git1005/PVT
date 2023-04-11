# PVT

This is a practice on phyramid vision transformer( PVT ).
It is based on the paper "[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction
without Convolutions](https://arxiv.org/pdf/2102.12122v2.pdf)"

# Architecture
![image](https://user-images.githubusercontent.com/63143667/231092644-6abebd60-bcab-4566-8a59-630a1589daaa.png)
```
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─ModuleList: 1                          []                        --
|    └─ModuleList: 2                     []                        --
|    |    └─Patch_Pos_embed: 3-1         [-1, 3136, 64]            203,904
|    |    └─block: 3-2                   [-1, 3136, 64]            1,035,584
|    └─ModuleList: 2                     []                        --
|    |    └─Patch_Pos_embed: 3-3         [-1, 784, 128]            133,376
|    |    └─block: 3-4                   [-1, 784, 128]            4,736,256
|    └─ModuleList: 2                     []                        --
|    |    └─Patch_Pos_embed: 3-5         [-1, 196, 320]            227,200
|    |    └─block: 3-6                   [-1, 196, 320]            44,349,760
|    └─ModuleList: 2                     []                        --
|    |    └─Patch_Pos_embed: 3-7         [-1, 49, 512]             681,984
├─LayerNorm: 1-1                         [-1, 49, 512]             1,024
├─ModuleList: 1                          []                        --
|    └─ModuleList: 2                     []                        --
|    |    └─block: 3-8                   [-1, 50, 512]             10,244,608
├─Sequential: 1-2                        [-1, 500]                 --
|    └─LayerNorm: 2-1                    [-1, 512]                 1,024
|    └─Linear: 2-2                       [-1, 500]                 256,500
==========================================================================================
Total params: 61,871,220
Trainable params: 61,871,220
Non-trainable params: 0
Total mult-adds (M): 62.47
==========================================================================================
Input size (MB): 0.57
Forward/backward pass size (MB): 0.20
Params size (MB): 236.02
Estimated Total Size (MB): 236.79
```
# Data
Image data is from kaggle "BIRDS 500 SPECIES- IMAGE CLASSIFICATION"(https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
## File structure
```
birds
  ├── test
  │   ├── ABBOTTS BABBLER
  |   ├── ABBOTTS BOOBY
  |   |         .
  |   |         .
  |   |         .
  |
  ├── train
  │   ├── ABBOTTS BABBLER
  │   ├── ABBOTTS BABBLER
  |   |         .
  |   |         .
  |   |         .
  |
  ├── val
  │   ├── ABBOTTS BABBLER
  │   ├── ABBOTTS BABBLER
  |   |         .
  |   |         .
  |   |         .
```

Inference: <br/>
data:<br/>
https://www.kaggle.com/code/jainamshah17/pytorch-starter-image-classification <br/>
https://www.kaggle.com/code/lonnieqin/bird-classification-with-pytorch <br/>
https://www.kaggle.com/code/stpeteishii/bird-species-classify-torch-conv2d <br/>
https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code?datasetId=534640&searchQuery=torch <br/>

ViT:<br/>
https://github.com/blackfeather-wang/Dynamic-Vision-Transformer
https://github.com/whai362/PVT/tree/57e2dfaa5a46f9050d76f306a4fcd9a7c061f520

tensoeboard:<br/>
https://github.com/HyoungsungKim/Pytorch-tensorboard_tutorial <br/>





