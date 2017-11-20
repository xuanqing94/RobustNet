#!/bin/bash

device=3
dataset=cifar10
root=~/data/cifar10-py
net=resnext
min_eps=0.1
max_eps=0.3
model_out=./resnext/adv_${min_eps}_${max_eps}.pth
log_file=./resnext/log_adv_${min_eps}_${max_eps}.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} ./main_adv.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --minEps ${min_eps} --maxEps ${max_eps} > ${log_file}
