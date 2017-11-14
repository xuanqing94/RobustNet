#!/bin/bash
dataset=cifar10
root=~/data/cifar10-py

net=resnext

attack=CW

noise_init=0
noise_inner=0
ensemble=1

defense=ours_${net}_${noise_init}_${noise_inner}_${ensemble}
c=0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01,0.03,0.04,0.06,0.08,0.1,0.12,0.14,0.2
#c=0.9
mode=test

srcModel=./${net}/noise_${noise_init}_${noise_inner}.pth
dstModel=./${net}/noise_${noise_init}_${noise_inner}.pth

CUDA_VISIBLE_DEVICES=3 ./attack.py --dataset ${dataset} --net ${net} --root ${root} --dstModel ${dstModel} --srcModel ${srcModel} --noiseInit ${noise_init} --noiseInner ${noise_inner} --c ${c} --attack ${attack} --mode ${mode} --ensemble ${ensemble} > ./experiment/${dataset}_${attack}_${defense}
