# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
# from engine import train_one_epoch, evaluate
# from losses import DistillDiffPruningLoss_dynamic
from samplers import RASampler
from augment import new_data_aug_generator
from calc_flops import throughput, get_flops

from contextlib import suppress

import models.vim
import models.videomamba

import utils

# log about
import mlflow


def get_args_parser():
    parser = argparse.ArgumentParser('DynamicVim training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='Vim-Tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-path', type=str, help='path to teacher model checkpoint')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'IMNET_100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=False)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # pruning params
    parser.add_argument('--base-rate', type=float, default=0.7)
    parser.add_argument('--path-keep-ratio', type=float, default=0.8)
    parser.add_argument('--token-ratio-weight', type=float, default=16.0)
    parser.add_argument('--path-ratio-weight', type=float, default=16.0)

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)

    parser.add_argument('--local-rank', default=0, type=int)
    return parser

def main(args):
    args.nb_classes = 1000

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    img_size = 224
    batch_size = 64
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    vim_tiny_baseline_path = "./pretrained_models/vim_t_midclstok_76p1acc.pth"
    vim_small_baseline_path = "./pretrained_models/vim_s_midclstok_80p5acc.pth"
    vim_base_baseline_path = "./pretrained_models/vim_b_midclstok_81p9acc.pth"
    
    # vim_tiny_0p7_0p7_path = ""
    # vim_tiny_0p7_0p8_path = "./output/comb-t-token0.7-path0.8/best_checkpoint.pth"
    # vim_tiny_0p8_0p7_path = "./output/comb-t-token0.8-path0.7/best_checkpoint.pth"
    # vim_tiny_0p8_0p8_path = "./output/comb-t-token0.8-path0.8/best_checkpoint.pth"
    
    
    
    vim_small_0p7_0p7_path = "./output/comb-s-token0.7-path0.7/best_checkpoint.pth"
    # vim_small_0p7_0p8_path = "./output/comb-s-token0.7-path0.8/best_checkpoint.pth"
    # vim_small_0p8_0p7_path = "./output/comb-s-token0.8-path0.7/best_checkpoint.pth"
    # vim_small_0p8_0p8_path = "./output/comb-s-token0.8-path0.8/best_checkpoint.pth"
    # vim_tiny_0p9_0p8_path = "./vim-t.pth"
    # vim_small_0p8_0p8_path = "./vim-s.pth"
    vim_base_0p7_0p7_path = "./vim-b.pth"
    # video_mamba_tiny_baseline_path = "./pretrained_models/videomamba_t16_in1k_res224.pth"
    # video_mamba_small_baseline_path = "./pretrained_models/videomamba_s16_in1k_res224.pth"
    # video_mamba_tiny_0p9_0p8_path = "./video-t.pth"
    # video_mamba_small_0p8_0p8_path = "./video-s.pth"
    
    models = [
        # (vim_tiny_baseline_path, "VimTinyTeacher"),
        (vim_small_baseline_path, "VimSmallTeacher"),
        (vim_base_baseline_path, "VimBaseTeacher"),
        # (vim_tiny_0p9_0p8_path, "VimTinyDiffPruning"),
        # (vim_small_0p8_0p8_path, "VimSmallDiffPruning"),
        # (vim_tiny_0p7_0p7_path, "VimTinyDiffPruning"),
        # (vim_tiny_0p7_0p8_path, "VimTinyDiffPruning"),
        # (vim_tiny_0p8_0p7_path, "VimTinyDiffPruning"),
        # (vim_tiny_0p8_0p8_path, "VimTinyDiffPruning"),
        (vim_small_0p7_0p7_path, "VimSmallDiffPruning"),
        # (vim_small_0p7_0p8_path, "VimSmallDiffPruning"),
        # (vim_small_0p8_0p7_path, "VimSmallDiffPruning"),
        # (vim_small_0p8_0p8_path, "VimSmallDiffPruning"),
        
        (vim_base_0p7_0p7_path, "VimBaseDiffPruning"),
        # (video_mamba_tiny_baseline_path, "VideoMambaTinyTeacher"),
        # (video_mamba_small_baseline_path, "VideoMambaSmallTeacher"),
        # (video_mamba_tiny_0p9_0p8_path, "VideoMambaTinyDiffPruning"),
        # (video_mamba_small_0p8_0p8_path, "VideoMambaSmallDiffPruning"),
    ]
    
    def calc_latency(model, batch_size=256, device="cuda"):

        model.eval()

        start = time.time()
        with torch.no_grad():
            for batch in tqdm(data_loader_val):
                images, _ = batch
                images = images.to(device)
                model(images)
        end = time.time()
        print(f"Latency of one batch: {(end - start) / len(data_loader_val)} seconds")
        print(f"Latency of all batches: {(end - start)} seconds")
        
    
    def build_model(model_param):

        ckpt_path = model_param[0]
        model_name = model_param[1]
        PRUNING_LOC = [6, 12, 18]
        print(model_name)
        if model_name.startswith("Vim"):
            if model_name.endswith("Teacher"):
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                )
            elif model_name.startswith("VimTiny"):
                base_rate = 0.9
                KEEP_RATE = [base_rate, base_rate**2, base_rate**3]
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                    token_ratio=KEEP_RATE,
                    pruning_loc=PRUNING_LOC
                )
            elif model_name.startswith("VimSmall"):
                base_rate = 0.7
                KEEP_RATE = [base_rate, base_rate**2, base_rate**3]
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                    token_ratio=KEEP_RATE,
                    pruning_loc=PRUNING_LOC
                )
            else:
                base_rate = 0.7
                KEEP_RATE = [base_rate, base_rate**2, base_rate**3]
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                    token_ratio=KEEP_RATE,
                    pruning_loc=PRUNING_LOC
                )
        else:
            if model_name.endswith("Teacher"):
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                )
            elif model_name.startswith("VideoMambaTiny"):
                base_rate = 0.9
                KEEP_RATE = [base_rate, base_rate**2, base_rate**3]
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                    token_ratio=KEEP_RATE,
                    pruning_loc=PRUNING_LOC
                )
            else:
                base_rate = 0.8
                KEEP_RATE = [base_rate, base_rate**2, base_rate**3]
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=args.nb_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    img_size=args.input_size,
                    token_ratio=KEEP_RATE,
                    pruning_loc=PRUNING_LOC
                )

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        utils.load_state_dict(model, checkpoint['model'])
        model.to(device)
        model.eval()
        return model
    
    for model_param in models:
        model = build_model(model_param)
        print(f'# Throughput test for {model_param[1]}')
        
        if "Small" in model_param[1]:
            throughput(model, device, batch_size=128, img_size=args.input_size)
        else:
            throughput(model, device, batch_size=64, img_size=args.input_size)
        # print(f'# Latency test for {model_param[1]}')
        # images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        # for _ in range(50):
        #     model(images)
        # torch.cuda.synchronize()
        
        # total_time = 0
        # start = time.time()
        # for batch in data_loader_val:
        #     images, labels = batch
        #     images = images.to(device, non_blocking=True)
        #     model(images)
        # torch.cuda.synchronize()
        # end = time.time()
        # total_time += (end - start)
        # print(f"Throughput: {batch_size * len(data_loader_val) / (end - start)} images/sec")
        
        
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)
