#!/bin/bash
mkdir -p ./logs
log_file="./logs/vim-s_$(date '+%Y-%m-%d_%H-%M-%S').log"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nproc_per_node=4 --master_port=15277 main.py --model Vim-Small --model-path /root/autodl-tmp/dyvm-vim-s.pth \
--data-set IMNET --data-path /root/autodl-tmp/imagenet --batch-size 64 --drop-path 0.0 --min-lr 1e-6 --lr 5e-5 --weight-decay 1e-8 --warmup-epochs 5 --num_workers 16 \
--token-keep-ratio 0.7 --block-keep-ratio 0.8 --token-pruning-weight 10.0 --block-pruning-weight 10.0 \
--eval \
--output_dir ./output/vim-s --no_amp 2>&1 | tee "$log_file"