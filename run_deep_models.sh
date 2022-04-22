#!/bin/bash
python deep_model.py --name FNC_FULL --arch fcn --data full
python deep_model.py --name FNC_NO_WATER --arch fcn --data no_water

python deep_model.py --name CNN_FULL --arch cnn --data full
python deep_model.py --name CNN_NO_WATER --arch cnn --data no_water

python deep_model.py --name CNN_DILATION_FULL --arch cnn_dilated --data full
python deep_model.py --name CNN_DILATION_NO_WATER --arch cnn_dilated --data no_water
