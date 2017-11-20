#!/bin/bash

device=2
dataset=stl10
root=~/data/stl10
net=stl10_model
T=100
model_teacher=./${net}/dd_teacher_${T}.pth
model_student=./${net}/dd_student_${T}.pth
log_file=./${net}/log_dd_${T}.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} ./main_dd.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_teacher} --root ${root} --role teacher --T ${T} > ${log_file}

CUDA_VISIBLE_DEVICES=${device} ./main_dd.py --lr ${lr} --dataset ${dataset} --net ${net} --modelIn ${model_teacher} --modelOut ${model_student} --root ${root} --role student --T ${T} >> ${log_file}

