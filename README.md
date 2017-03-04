# keras-voxresnet
A keras re-implementation of VoxResNet (Hao Chen et.al) for volumetric image classification. (Non-official)

keras-voxresnet enables __volumetric image classification__ with keras and tensorflow/theano.

Note that the original VoxResNet architecture was designed for volumetric image segmentation which is not yet implemented here.

The implementation is built upon the keras implementation of resnet  [keras-resnet](https://github.com/raghakot/keras-resnet).


### Installation

This implementation is based on Keras together with the GPU version of Tensorflow/Theano. It is highly recommended to run the training processes with a GPU-enabled Docker image.

The following installation procedures assumes Nvidia Driver, [docker](https://docs.docker.com/engine/installation/linux/ubuntu/) and [nvidia-docker](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/) are properly installed on a [Ubuntu machine](https://www.ubuntu.com/download/desktop/install-ubuntu-desktop).

First clone the repository:

```
$ git clone https://github.com/JihongJu/keras-voxresnet.git
```

Start bash in a Keras docker image:

```
$ nvidia-docker run -it --rm -v `pwd`/keras-voxresnet/:/workspace jihong/nvidia-keras bash
```

Install dependencies with `pip`:

```
# pip install -r requirements.txt
```


Validate installation:

```
/workspace# py.test tests
```

### Usage
