model=vgg16
noise=0.3
CUDA_VISIBLE_DEVICES=1 ./main2.py --lr 0.1 --net ${model} --noise ${noise} --modelOut ./${model}/noise_${noise}.pth > ./${model}/log_noise_${noise}.txt

