#!/bin/bash

device=2
dataset=cifar10
root=~/data/cifar10-py
net=resnext
init=0.4
inner=0.1
model_out=./${net}/rse_${init}_${inner}.pth
log_file=./${net}/log_rse_${init}_${inner}.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} ./main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}
