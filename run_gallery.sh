#!/bin/bash
device=3
dataset=cifar10
root=~/data/cifar10-py
net=resnext
defense=rse
model_in=./${dataset}_${net}/rse_0.1_0.1.pth
c=100
noise_init=0.1
noise_inner=0.1
ensemble=50
imagef='images'

CUDA_VISIBLE_DEVICES=${device} ./gallery.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --ensemble ${ensemble} --imgf ${imagef}

