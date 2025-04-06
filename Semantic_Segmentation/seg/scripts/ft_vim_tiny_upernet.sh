#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p output

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_160k.py
PRETRAIN_CKPT=/path/to/your/pretrained/checkpoint.pth

python apex.py

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=10295 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir output/vimseg-t-clip --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT}