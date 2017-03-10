"""A vanilla 3D resnet based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""
import six
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
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block (copy-paster from
    raghakot/keras-resnet)
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1,
                             kernel_dim2=kernel_dim2, kernel_dim3=kernel_dim3,
                             subsample=subsample, init=init,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f


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
            W_regularizer=l2(1e-4)
            )(input)

    return merge([shortcut, residual], mode="sum")


def _residual_block3d(block_function, nb_filter, repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 2)
            input = block_function(nb_filter=nb_filter,
                                   init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1, 1),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution3D(nb_filter=nb_filter,
                                  kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                                  subsample=init_subsample,
                                  init="he_normal", border_mode="same",
                                  W_regularizer=l2(0.0001)
                                  )(input)
        else:
            conv1 = _bn_relu_conv3d(nb_filter=nb_filter,
                                    kernel_dim1=3, kernel_dim2=3,
                                    kernel_dim3=3,
                                    subsample=init_subsample
                                    )(input)

        residual = _bn_relu_conv3d(nb_filter=nb_filter,
                                   kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                                   )(conv1)
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


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs,  block_fn, repetitions):
        """Instantiate a vanilla ResNet3D keras model
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (nb_filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """

        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple "
                            "(conv_dim1, conv_dim2, conv_dim3, channels) "
                            "for tensorflow as backend or "
                            "(channels, conv_dim1, conv_dim2, conv_dim3) "
                            "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu3D(nb_filter=64, kernel_dim1=7, kernel_dim2=7,
                                kernel_dim3=7, subsample=(2, 2, 2)
                                )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                             border_mode="same")(conv1)

        # repeat blocks
        block = pool1
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, nb_filter=nb_filter,
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            nb_filter *= 2

        # last activation
        block = _bn_relu(block)
        block_norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)


        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                            block._keras_shape[DIM2_AXIS],
                                            block._keras_shape[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        if num_outputs > 1:
            dense = Dense(output_dim=num_outputs, init="he_normal",
                          activation="softmax")(flatten1)
        else:
            dense = Dense(output_dim=num_outputs, init="he_normal",
                          activation="sigmoid")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3])
