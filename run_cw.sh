#!/bin/bash
device=0
dataset=cifar10
root=~/data/cifar10-py
net=vgg16
defense=brelu
model_in=./${net}/brelu_0.05.pth
c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
noise_init=0
noise_inner=0
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}.acc

CUDA_VISIBLE_DEVICES=${device} ./cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}
