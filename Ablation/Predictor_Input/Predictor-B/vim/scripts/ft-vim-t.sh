#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model Vim-Tiny --model-path /root/autodl-tmp/vim_t_midclstok_76p1acc.pth --batch-size 128 --num_workers 25 --data-set IMNET --data-path /root/autodl-tmp/imagenet --output_dir ./output/ft-vim-t --no_amp 2>&1 | tee ./train_dyvim.log
