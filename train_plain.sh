#!/bin/bash

device=0
dataset=cifar10
root=~/data/cifar10-py
net=resnext
model_out=./resnext/plain.pth
log_file=./resnext/log_plain.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} ./main_plain.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} > ${log_file}
