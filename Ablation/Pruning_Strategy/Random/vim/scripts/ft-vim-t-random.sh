#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model Vim-Tiny --model-path /mnt/data-1/users/work/vim_t_midclstok_76p1acc.pth --batch-size 128 --lr 5e-5 --weight-decay 1e-8 --num_workers 10 --data-set IMNET --data-path /mnt/data-1/users/work/ImageNet --output_dir ./output/ft-vim-t-random --no_amp 2>&1 | tee ./train_dyvim.log
