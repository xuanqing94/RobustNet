#!/bin/bash

device=2
dataset=cifar10
root=~/data/cifar10-py
net=resnext
init=0.05
model_out=./resnext/brelu_${init}.pth
log_file=./resnext/log_brelu_${init}.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} ./main_brelu.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} > ${log_file}
