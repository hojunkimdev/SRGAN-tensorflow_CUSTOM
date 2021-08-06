#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./data/inference/ \
    --summary_dir ./data/inference/log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --batch_size 16 \
    --input_dir_LR  /home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Train_Val_ksh/scan/crop/Train/ALL \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint /home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/experiment_SRGAN_VGG54/ver11_cyclegan_colroHR_train/model-200000