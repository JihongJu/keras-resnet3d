# keras-resnet3d

[![Build Status](https://travis-ci.org/JihongJu/keras-resnet3d.svg?branch=master)](https://travis-ci.org/JihongJu/keras-resnet3d) [![codecov](https://codecov.io/gh/JihongJu/keras-resnet3d/branch/master/graph/badge.svg)](https://codecov.io/gh/JihongJu/keras-resnet3d)


### Resnet 3D


A vanilla 3D extention to [raghakot/keras-resnet](https://github.com/raghakot/keras-resnet)



### VoxResNet (TODO)
A keras re-implementation of VoxResNet (Hao Chen et.al) for volumetric image segmention. (Non-official)

keras-voxresnet enables __volumetric image classification__ with keras and tensorflow/theano.


### Installation

#### Dependencies

[keras](https://keras.io/#installation), [tensorflow](https://www.tensorflow.org/install/)/[theano](http://deeplearning.net/software/theano/install.html) and their corresponding dependencies.


#### Install with `pip`

```bash
$ pip install git+https://github.com/JihongJu/keras-resnet3d.git
```


#### Build from source

```bash
$ git clone https://github.com/JihongJu/keras-resnet3d.git
$ cd keras-resnet3d
$ python setup.py build
```

### Usage

```python
from resnet3d import Resnet3DBuilder
model = Resnet3DBuilder.build_resnet_50((96, 96, 96, 1), 20)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32)
```

More details see example:

 - [A  lung-cancer-detector](https://github.com/JihongJu/lung-cancer-detector/tree/master/resnet3d_50)
