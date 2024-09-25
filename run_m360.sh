#!/bin/bash

CUDA_VISIBLE_DEVICES=0
DATA_DIR=/mnt/Data2/nerf_datasets/m360/

python train_w_viewer.py -s $DATA_DIR/garden -m output/m360/garden

# To view the results：
# 2D-GS-Viser-Viewer
# forward the output to a target port and do ssh mapping
# ssh -L 8080:127.0.0.1:8080 lchen39@155.246.81.47
# conda activate splatam
# python viewer.py /mnt/Data2/liyan/2d-gaussian-splatting/output/nerf_synthetic/chair/point_cloud/iteration_30000/point_cloud.ply -s /mnt/Data2/nerf_datasets/nerf_synthetic/chair/
