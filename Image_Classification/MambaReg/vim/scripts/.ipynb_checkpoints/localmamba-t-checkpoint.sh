#!/bin/bash
mkdir -p ./logs
log_file="./logs/localmamba-t_$(date '+%Y-%m-%d_%H-%M-%S').log"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nproc_per_node=4 --master_port=15277 main.py --model LocalMamba-Tiny --model-path ./pretrained_models/local_vim_tiny.ckpt \
--data-set IMNET --data-path /root/autodl-tmp/imagenet --batch-size 48 --drop-path 0.0 --min-lr 1e-6 --lr 3e-5 --weight-decay 1e-8 --warmup-epochs 5 --num_workers 16 \
--token-keep-ratio 0.9 --block-keep-ratio 0.8 --token-pruning-weight 20.0 --block-pruning-weight 20.0 \
--output_dir ./output/localmamba-t --no_amp 2>&1 | tee "$log_file"