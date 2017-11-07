model=vgg19
noise=0.7
CUDA_VISIBLE_DEVICES=2 ./main2.py --lr 0.1 --noise ${noise} --modelOut ./${model}/noise_${noise}.pth > ./${model}/log_noise_${noise}.txt

