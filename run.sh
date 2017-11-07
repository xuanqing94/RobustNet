model=resnetxt
noise=0.1
CUDA_VISIBLE_DEVICES=1 ./main2.py --lr 0.1 --noise ${noise} --modelOut ./${model}/noise_${noise}.pth > ./${model}/log_noise_${noise}.txt

