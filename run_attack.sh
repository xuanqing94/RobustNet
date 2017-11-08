#!/bin/bash

CUDA_VISIBLE_DEVICES=3 ./attack.py --root ~/data/cifar10-py --modelIn ./vgg16/noise_0.05_0.1.pth --noiseInit 0. --noiseInner 0. --c 0.5
