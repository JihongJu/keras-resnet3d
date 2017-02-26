"""A VoxResNet implementation for keras

This module imaplement the VoxResNet with 17 conv layers
on top of Raghavendra Kotikalapudi's residual networks implementation keras-resnet.
See https://github.com/raghakot/keras-resnet.
"""

from keras.models import Model
from keras import backend as K
from resnet.resnet import _handle_dim_ordering, _bn_relu


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block.
    """
    pass


def voxres_block(nb_filter):
    """VoxRes architecture
    See https://arxiv.org/pdf/1608.05895.pdf
    """
    pass


class VoxResnetBuilder(object):
    @staticmethod
    def build_classification(input_shape, num_outputs):
        """Instantiate a keras model for VoxResNet
        # Arguments
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
            num_outputs: The number of outputs at the
        """

        _handle_dim_ordering()
        pass

    @staticmethod
    def build_segmentation(input_shape, num_outputs):
        """
        """
        pass
