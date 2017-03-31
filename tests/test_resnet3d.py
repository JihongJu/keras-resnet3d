"""Test resnet3d."""
import pytest
from keras import backend as K
from resnet3d import Resnet3DBuilder


@pytest.fixture
def resnet3d_test():
    """resnet3d test helper."""
    def f(model):
        K.set_image_data_format('channels_last')
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to build with {}".format(K.image_data_format())
    return f


def test_resnet3d_18(resnet3d_test):
    """Test 18."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_18((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_18((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_34(resnet3d_test):
    """Test 34."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_34((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_34((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_50(resnet3d_test):
    """Test 50."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_50((224, 224, 224, 1), 1, 1e-2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_50((1, 512, 512, 256), 1, 1e-2)
    resnet3d_test(model)


def test_resnet3d_101(resnet3d_test):
    """Test 101."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_101((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_101((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_resnet3d_152(resnet3d_test):
    """Test 152."""
    K.set_image_data_format('channels_last')
    model = Resnet3DBuilder.build_resnet_152((224, 224, 224, 1), 2)
    resnet3d_test(model)
    K.set_image_data_format('channels_first')
    model = Resnet3DBuilder.build_resnet_152((1, 512, 512, 256), 2)
    resnet3d_test(model)


def test_bad_shape():
    """Input shape need to be 4."""
    K.set_image_data_format('channels_last')
    with pytest.raises(ValueError):
        Resnet3DBuilder.build_resnet_152((224, 224, 224), 2)


def test_get_block():
    """Test get residual block."""
    K.set_image_data_format('channels_last')
    Resnet3DBuilder.build((224, 224, 224, 1), 2, 'bottleneck',
                          [2, 2, 2, 2], reg_factor=1e-4)
    assert True
    with pytest.raises(ValueError):
        Resnet3DBuilder.build((224, 224, 224, 1), 2, 'nullblock',
                              [2, 2, 2, 2], reg_factor=1e-4)
