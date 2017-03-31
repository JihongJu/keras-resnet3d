#!/usr/bin/env bash

cd ${HOME}/workspace/
virtualenv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt --upgrade
python setup.py build
