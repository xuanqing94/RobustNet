# Stage I. step size = 0.1
#CUDA_VISIBLE_DEVICES=0 ./main2.py --lr 0.00625 --noise 0.3 --modelIn noise0.3_IV.pth --modelOut noise0.3_V.pth > log_noise_0.3_V.txt &
CUDA_VISIBLE_DEVICES=1 ./main2.py --lr 0.025 --noise 0.01 --modelIn ./vgg16/noise0.01_II.pth --modelOut ./vgg16/noise0.01_III.pth > ./vgg16/log_noise_0.01_III.txt &
CUDA_VISIBLE_DEVICES=2 ./main2.py --lr 0.025 --noise 0.1 --modelIn ./vgg16/noise0.1_II.pth --modelOut ./vgg16/noise0.1_III.pth > ./vgg16/log_noise_0.1_III.txt &
CUDA_VISIBLE_DEVICES=3 ./main2.py --lr 0.025 --noise 0 --modelIn ./vgg16/noise0_II.pth --modelOut ./vgg16/noise0_III.pth > ./vgg16/log_noise_0_III.txt &
CUDA_VISIBLE_DEVICES=3 ./main2.py --lr 0.025 --noise 0.5 --modelIn ./vgg16/noise0.5_II.pth --modelOut ./vgg16/noise0.5_III.pth > ./vgg16/log_noise_0.5_III.txt &
#CUDA_VISIBLE_DEVICES=2 ./main2.py --lr 0.1 --noise 0.5 --modelIn noise0.5_III.pth --modelOut noise0.5_IV.pth > log_noise_0.5_IV.txt

