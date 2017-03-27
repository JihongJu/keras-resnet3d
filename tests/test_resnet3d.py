import pytest
from keras import backend as K
from resnet3d import Resnet3DBuilder


def test_resnet3d_18():
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_18((224, 224, 224, 1), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with tensorflow"

    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_18((1, 512, 512, 256), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with theano"


def test_resnet3d_34():
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_34((224, 224, 224, 1), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with tensorflow"

    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_34((1, 512, 512, 256), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with theano"


def test_resnet3d_50():
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_50((224, 224, 224, 1), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with tensorflow"

    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_50((1, 512, 512, 256), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with theano"
