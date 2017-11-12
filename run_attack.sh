#!/bin/bash
architect=vgg16
dataset=cifar10
attack=CW

noise_init=0.6
noise_inner=0.2
ensemble=1

defense=ours_${noise_init}_${noise_inner}_${ensemble}
c=0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
mode=test

#CUDA_VISIBLE_DEVICES=0 ./attack.py --root ~/data/cifar10-py --modelIn ./vgg16/noise_0.6_0.1.pth --noiseInit 0.6 --noiseInner 0.1 --c 0.01 --mode peek #> ./experiment/${dataset}_${attack}_${defense}
CUDA_VISIBLE_DEVICES=1 ./attack.py --root ~/data/cifar10-py --modelIn ./${architect}/noise_${noise_init}_${noise_inner}.pth --noiseInit ${noise_init} --noiseInner ${noise_inner} --c ${c} --attack ${attack} --mode ${mode} --ensemble ${ensemble} > ./experiment/${dataset}_${attack}_${defense}
