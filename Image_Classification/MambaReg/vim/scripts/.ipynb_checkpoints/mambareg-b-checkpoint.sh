#!/bin/bash
mkdir -p ./logs
log_file="./logs/mambareg-b_$(date '+%Y-%m-%d_%H-%M-%S').log"
export CUDA_VISIBLE_DEVICES="0"
torchrun --nproc_per_node=1 --master_port=15277 main.py --model MambaReg-Base --model-path ./pretrained_models/mambar_base_patch16_224.pth \
--data-set IMNET --data-path /root/autodl-tmp/imagenet --batch-size 32 --drop-path 0.0 --min-lr 1e-6 --lr 5e-5 --weight-decay 1e-8 --warmup-epochs 5 --num_workers 16 \
--token-keep-ratio 0.7 --block-keep-ratio 0.7 --token-pruning-weight 10.0 --block-pruning-weight 10.0  \
--output_dir ./output/mambareg-b --no_amp 2>&1 | tee "$log_file"