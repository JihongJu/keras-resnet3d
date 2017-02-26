"""A VoxResNet implementation for keras

This module implement the VoxResNet with 17 conv layers based on
Raghavendra Kotikalapudi's residual networks implementation keras-resnet.
See https://github.com/raghakot/keras-resnet.
"""

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution3D,
    AveragePooling3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
# from resnet.resnet import _bn_relu


def _bn_relu(input):
    """Helper to build a BN -> relu block (copy-paster from
    raghakot/keras-resnet)
    """
    norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block.
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1,
                             kernel_dim2=kernel_dim2, kernel_dim3=kernel_dim3,
                             subsample=subsample, init=init,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum" at
    the same time
    """
    stride_dim1 = input._keras_shape[DIM1_AXIS] \
        // residual._keras_shape[DIM1_AXIS]
    stride_dim2 = input._keras_shape[DIM2_AXIS] \
        // residual._keras_shape[DIM2_AXIS]
    stride_dim3 = input._keras_shape[DIM3_AXIS] \
        // residual._keras_shape[DIM3_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Convolution3D(
            nb_filter=residual._keras_shape[CHANNEL_AXIS],
            kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
            subsample=(stride_dim1, stride_dim2, stride_dim3),
            init="he_normal", border_mode="valid",
            W_regularizer=l2(1e-4))(input)

    return merge([shortcut, residual], mode="sum")


def _voxel_residual_block(nb_filter, repetitions):
    """Builds a residual block with repeating _bn_relu_conv3d /2 > voxres block
    # Arguments
        nb_filter: number of filters, int8
        repetitions: number of repetitions, int8
    Returns:
        output: Tuple of outputs of size repetitions + 1
        e.g. (input, output_rep1, output_rep2,  output_rep3) if repetitions = 3
    """
    def f(input):
        output = (input,)
        for i in range(repetitions):
            conv3_3_3_s2 = _bn_relu_conv3d(nb_filter=nb_filter, kernel_dim1=3,
                                           kernel_dim2=3, kernel_dim3=3,
                                           subsample=(2, 2, 2))(input)
            input = voxres(nb_filter=nb_filter)(conv3_3_3_s2)
            output += (input,)
        return output

    return f


def voxres(nb_filter):
    """VoxRes module with two 64, 3x3x3 conv layers and skip connection
    See https://arxiv.org/pdf/1608.05895.pdf
    """
    def f(input):
        conv3_3_3 = _bn_relu_conv3d(nb_filter=nb_filter, kernel_dim1=3,
                                    kernel_dim2=3, kernel_dim3=3)(input)
        residual = _bn_relu_conv3d(nb_filter=nb_filter, kernel_dim1=3,
                                   kernel_dim2=3, kernel_dim3=3)(conv3_3_3)
        return _shortcut3d(input, residual)

    return f


def _handle_dim_ordering():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


class VoxResnetBuilder(object):
    @staticmethod
    def build_classification(input_shape, num_outputs):
        """Instantiate a keras model for VoxResNet
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (nb_filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer.
        # Returns
            model: a VoxResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """

        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple "
                            "(conv_dim1, conv_dim2, conv_dim3, channels) "
                            "for tensorflow as backend or "
                            "(channels, conv_dim1, conv_dim2, conv_dim3) "
                            "for theano as backend")

        input = Input(shape=input_shape)
        conv1a = Convolution3D(nb_filter=32, kernel_dim1=3, kernel_dim2=3,
                               kernel_dim3=3, init="he_normal",
                               border_mode="same",
                               W_regularizer=l2(1.e-4))(input)
        conv1b = _bn_relu_conv3d(nb_filter=32, kernel_dim1=3, kernel_dim2=3,
                                 kernel_dim3=3, subsample=(1, 1, 1))(conv1a)
        voxres9 = _voxel_residual_block(nb_filter=64,
                                        repetitions=3)(conv1b)[-1]

        # Last activation
        block = _bn_relu(voxres9)
        block_norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        # Classifier block
        pool1 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                            block._keras_shape[DIM2_AXIS],
                                            block._keras_shape[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool1)
        dense = Dense(output_dim=num_outputs, init="he_normal",
                      activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_segmentation(input_shape, num_outputs):
        """TODO
        """
        pass
