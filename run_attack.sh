#!/bin/bash
CUDA_VISIBLE_DEVICES=3 ./attack.py --root ~/data/cifar10-py --modelIn ./vgg16/noise_0_0.pth --noiseInit 0 --noiseInner 0 --c 0.1
