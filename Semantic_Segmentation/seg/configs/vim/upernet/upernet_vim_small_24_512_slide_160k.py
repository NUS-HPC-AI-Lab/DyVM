# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
_base_ = [
    '../../_base_/models/upernet_vim.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='VisionMambaSeg',
        img_size=512, 
        patch_size=16, 
        in_chans=3,
        embed_dim=384, 
        depth=24,
        out_indices=[5, 11, 17, 23],
        pretrained=None,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        # if_bimamba=True,
        bimamba_type="v2",
        final_pool_type='all',
        if_divide_out=True,
        if_cls_token=False,
        pruning_loc=[6, 12, 18],
        token_ratio=[0.8, 0.8**2, 0.8**3],
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        num_classes=150,
        channels=384,
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=12e-5,  # Updated initial learning rate
    betas=(0.9, 0.999), 
    weight_decay=0.01,  # Updated weight decay
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.92)
)

lr_config = dict(
    _delete_=True, 
    policy='poly',  # Keep poly or change to 'linear' if strict adherence is required
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

# By default, models are trained on 4 GPUs with 8 images per GPU
data=dict(samples_per_gpu=16, workers_per_gpu=8)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=dict(max_norm=5, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)