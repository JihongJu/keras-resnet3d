#!/usr/bin/env python

from setuptools import setup

setup(name='keras-resnet3d',
      version='0.0.1',
      description='Keras implementation of VoxResNet by Hao Chen et.al.',
      author='Jihong Ju',
      author_email='daniel.jihong.ju@gmail.com',
      install_requires=['keras>=2.0.0'],
      packages=['resnet3d'])
