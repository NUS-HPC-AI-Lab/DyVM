#!/bin/bash
mkdir -p ./logs
log_file="./logs/videomamba-t_$(date '+%Y-%m-%d_%H-%M-%S').log"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nproc_per_node=4 --master_port=15277 main.py --model VideoMamba-Tiny --model-path ./pretrained_models/videomamba_t16_in1k_res224.pth \
--data-set IMNET --data-path /path/to/your/imagenet --batch-size 128 --drop-path 0.0 --min-lr 1e-6 --lr 3e-5 --weight-decay 1e-8 --warmup-epochs 5 --num_workers 16 \
--token-keep-ratio 0.9 --block-keep-ratio 0.8 --token-pruning-weight 10.0 --block-pruning-weight 10.0 \
--output_dir ./output/videomamba-t --no_amp 2>&1 | tee "$log_file"