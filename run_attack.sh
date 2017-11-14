#!/bin/bash
architect=vgg16
dataset=cifar10
attack=RAND_FGSM

noise_init=0
noise_inner=0
ensemble=1

defense=ours_${noise_init}_${noise_inner}_${ensemble}
#c=0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
c=0.01,0.03,0.04,0.06,0.08,0.1,0.12,0.14,0.2
#c=0.9
mode=test

srcModel=./${architect}/noise_${noise_init}_${noise_inner}.pth
dstModel=./${architect}/noise_${noise_init}_${noise_inner}.pth

CUDA_VISIBLE_DEVICES=2 ./attack.py --root ~/data/cifar10-py --dstModel ${dstModel} --srcModel ${srcModel} --noiseInit ${noise_init} --noiseInner ${noise_inner} --c ${c} --attack ${attack} --mode ${mode} --ensemble ${ensemble} > ./experiment/${dataset}_${attack}_${defense}
