import pytest
from keras import backend as K
from voxresnet import VoxResnetBuilder


def test_voxresnet():
    K.set_image_dim_ordering('tf')
    model = VoxResnetBuilder.build_classification((512, 512, 256, 1), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with tensorflow"

    K.set_image_dim_ordering('th')
    model = VoxResnetBuilder.build_classification((1, 512, 512, 256), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with theano"
