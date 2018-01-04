# SSD: Single Shot MultiBox Object Detector
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)
from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang,
and Alexander C. Berg.

The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

This is a forked version of the repo [here](https://github.com/amdegroot/ssd.pytorch).
Many thanks to the original author [@Max](https://github.com/amdegroot)!

<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>References</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/), clone this repository, etc.
- Please follow the instructions in the [original](https://github.com/amdegroot/ssd.pytorch#installation) forked version.
- We use [wisdom](https://github.com/facebookresearch/visdom) to visualize results and training process.
There is a tutorial on how to use it.

- (Optional, buggy now) install `plotly`: `conda install -c plotly plotly`

## Datasets
To make things easy, we provide a simple dataset loader that enherits `torch.utils.data.Dataset`
making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC 2007 trainval & test and 2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
sh data/scripts/VOC2012.sh # <directory>
```

### COCO

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:
https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `data/pretrain` dir:

```Shell
mkdir data/pretrain
cd data/pretrain
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script,
simply specify the parameters listed in `option/train_opt.py`.

```Shell
python train.py
```

<!---
- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Currently we only support training on v2 (the newest version).
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)
-->

## Evaluation/Test
To evaluate a trained network:

```Shell
python test.py
```
You can specify the parameters listed in the `option/test_opt.py` file by flagging them or manually changing them.



## Performance

### VOC2007 Test

#### mAP

#### FPS
**GTX 1060:** ~45.45 FPS 

## Demo (Grab and Go)

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models: 
    * SSD300 v2 trained on VOC0712 (newest PyTorch version)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 v2 trained on VOC0712 (original Caffe version)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
    * SSD300 v1 (original/old pool6 version) trained on VOC07
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_voc07.tar.gz
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325) 
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run): 
    `jupyter notebook` 

    2. If using [pip](https://pypi.python.org/pypi/pip):
    
```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## TODO
We have accumulated the following to-do list, which you can expect to be done in the very near future
- Still to come:
  * Train SSD300 with batch norm
  * Add support for SSD512 training and testing
  * Add support for COCO dataset
  * Create a functional model definition for Sergey Zagoruyko's [functional-zoo](https://github.com/szagoruyko/functional-zoo)


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo): 
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow) 
