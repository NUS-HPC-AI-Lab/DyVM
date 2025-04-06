import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba, DynamicMamba


@torch.no_grad()
def throughput(model, device, batch_size, img_size=224):
    model.eval()

    images = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # warmup
    for _ in range(50):
        model(images)

    torch.cuda.synchronize()
    tic1 = time.time()

    for _ in range(30):
        model(images)

    torch.cuda.synchronize()
    tic2 = time.time()

    print(f"Throughput: {batch_size * 30 / (tic2 - tic1)} images/sec")

def get_flops(model, img_size=224, show_detail=False):
    conv_flops = []

    def conv_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        input_tensor = input[0]
        output_tensor = output[0]
        batch_size, input_channels, input_height, input_width = input_tensor.size()
        output_channels, output_height, output_width = output_tensor.size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        conv_flops.append(flops)
        if show_detail:
            print(f'Conv2d flops: {flops}')

    linear_flops = []

    def linear_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        input_tensor = input[0]
        in_feature, out_feature = self.weight.size()
        batch_size = input_tensor.numel() // input_tensor.size(-1)
        flops = batch_size * in_feature * out_feature

        linear_flops.append(flops)
        if show_detail:
            print(f'Linear flops: {flops}')

    mamba_flops = []

    def mamba_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        """
        Glossary:
        b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
        l: sequence length                  (`L` in [1] Algorithm 2)
        d or d_model: hidden dim
        n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        expand: expansion factor            (`E` in [1] Section 3.4)
        d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
        A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
        Δ or delta: input-dependent step size
        dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
        """

        input_tensor = input[0]
        b, l, _ = input_tensor.size()

        flops = 0

        block_keep_ratio = self.block_keep_ratio if hasattr(self, 'block_keep_ratio') else 1.0

        # 1. in_proj
        flops += b * l * self.d_model * self.d_inner * 2
        # 2.1 causual conv1d
        flops += b * (l + self.d_conv - 1) * self.d_inner * self.d_conv * block_keep_ratio
        # 2.2 x_proj
        flops += b * l * self.d_inner * (self.dt_rank + self.d_state * 2) * block_keep_ratio
        # 2.3 dt_proj
        flops += b * l * self.dt_rank * self.d_inner * block_keep_ratio
        # 3 selective scan
        # https://github.com/state-spaces/mamba/issues/110
        flops += (9 * b * l * self.d_inner * self.d_state + 2 * b * l * self.d_inner) * block_keep_ratio
        # 4 out_proj
        flops += b * l * self.d_inner * self.d_model

        mamba_flops.append(flops)
        if show_detail:
            print(f'Mamba flops: {flops}')

    def register_module_hook(net, hook_handle, prefix='', path=''):
        for name, module in net.named_children():
            registerd = False
            if isinstance(module, torch.nn.Conv2d):
                hook_handle.append(module.register_forward_hook(lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: conv_hook(*args, path=path)))
                registerd = True
            if isinstance(module, torch.nn.Linear):
                hook_handle.append(module.register_forward_hook(lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: linear_hook(*args, path=path)))
                registerd = True
            if isinstance(module, Mamba) or isinstance(module, DynamicMamba):
                hook_handle.append(module.register_forward_hook(lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: mamba_hook(*args, path=path)))
                registerd = True
            
            if registerd:
                if show_detail:
                    print(f"{prefix}{name}: {module.__class__.__name__} (registerd)")
            else:
                if show_detail:
                    print(f"{prefix}{name}: {module.__class__.__name__}")
                register_module_hook(module, hook_handle, prefix + '  ', path + '/' + name + ':' + module.__class__.__name__)

    hooks = []
    register_module_hook(model, hooks)

    input_shape = (3, img_size, img_size)
    input = torch.rand(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(input)
    for handle in hooks:
        handle.remove()

    total_flops = sum(sum(i) for i in [conv_flops, linear_flops, mamba_flops])
    return total_flops
