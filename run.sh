dataset=cifar10
root=~/data/cifar10-py
model=resnext
noise_init=0.9
noise_inner=0.2
#CUDA_VISIBLE_DEVICES=0 ./main2.py --lr 0.1 --root ${root} --net ${model} --noiseInit ${noise_init} --noiseInner ${noise_inner} --modelIn ./${model}/noise_${noise_init}_${noise_inner}.pth --modelOut ./${model}/noise_${noise_init}_${noise_inner}_adv.pth --adv > ./${model}/log_noise_${noise_init}_${noise_inner}_adv.txt
CUDA_VISIBLE_DEVICES=2 ./main2.py --lr 0.1 --dataset ${dataset} --root ${root} --net ${model} --noiseInit ${noise_init} --noiseInner ${noise_inner} --modelOut ./${model}/noise_${noise_init}_${noise_inner}.pth > ./${model}/log_noise_${noise_init}_${noise_inner}.txt

