# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import Mlp, DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from utils import get_cls_idx, _init_weights, segm_init_weights, extract_cls_and_other, combine_cls_and_other, rearrange_pos, batch_index_select

from models.embedding import PatchEmbed
from models.mamba_block import create_block, create_block_dynamic
from models.predictor import PredictorLG

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class DyVM(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16,
                 depth=24, 
                 embed_dim=192,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 num_cls_tokens=1,
                 cls_reduce=1,
                 pruning_loc=None, 
                 token_ratio=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.num_cls_tokens = num_cls_tokens
        self.cls_reduce = cls_reduce

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
            self.pos_embed_cls = nn.Parameter(
                torch.zeros(1, num_cls_tokens, self.embed_dim))
            H, W = self.patch_embed.grid_size
            self.token_idx, self.cls_positions = get_cls_idx(H, W, num_cls_tokens)
            
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if cls_reduce > 1:
            self.neck = nn.Linear(self.num_features, self.num_features // cls_reduce, bias=False)
            self.norm_neck = (nn.LayerNorm if not rms_norm else RMSNorm)(
                embed_dim * num_cls_tokens // cls_reduce, eps=norm_epsilon, **factory_kwargs)
            
        if num_classes < 1:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.num_features * (num_cls_tokens // cls_reduce), num_classes)

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block_dynamic(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_divide_out=True,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.score_predictors = nn.ModuleList(
            [
                PredictorLG(embed_dim) for _ in range(len(pruning_loc))
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        if cls_reduce > 1:
            self.neck.apply(segm_init_weights)            
        if self.num_cls_tokens > 0:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.pos_embed_cls, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "pos_embed_cls"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token       
        x = self.patch_embed(x)
        B, M, _ = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.num_cls_tokens > 0:
            cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
            x = torch.cat([x, cls_token], dim=1)[:, self.token_idx]

        cls_token_position = self.cls_positions.unsqueeze(0).expand(B, -1)
        assert cls_token_position.shape == (B, self.num_cls_tokens)

        # mamba impl
        residual = None
        hidden_states = x

        predictor_count = 0
        pred_decisions = []
        block_policies = []
        current_pos = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)
        token_policy = torch.ones((B, M, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        mask = torch.ones((B, M + self.num_cls_tokens, 1), dtype=hidden_states.dtype, device=hidden_states.device)

        for n, layer in enumerate(self.layers):  

            if n in self.pruning_loc:
                if self.training:
                    cls_t, other_t = extract_cls_and_other(hidden_states, cls_token_position, num_cls_token=self.num_cls_tokens)
                    cls_r, other_r = extract_cls_and_other(residual, cls_token_position, num_cls_token=self.num_cls_tokens)

                    pred_score = self.score_predictors[predictor_count](other_t + other_r, token_policy)
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * token_policy
                    pred_decisions.append(hard_keep_decision.squeeze(-1))
                    hard_keep_decision, other_t, other_r, current_pos, cls_token_position = rearrange_pos(hard_keep_decision, other_t, other_r, current_pos, num_cls_token=self.num_cls_tokens)
                    
                    hidden_states = combine_cls_and_other(cls_t, other_t, cls_token_position, num_cls_token=self.num_cls_tokens)
                    residual = combine_cls_and_other(cls_r, other_r, cls_token_position, num_cls_token=self.num_cls_tokens)
                    mask = combine_cls_and_other(torch.ones((B, self.num_cls_tokens, 1), dtype=hard_keep_decision.dtype, device=hard_keep_decision.device), hard_keep_decision, cls_token_position, num_cls_token=self.num_cls_tokens)
                    token_policy = hard_keep_decision
                else:
                    cls_t, other_t = extract_cls_and_other(hidden_states, cls_token_position, num_cls_token=self.num_cls_tokens)
                    cls_r, other_r = extract_cls_and_other(residual, cls_token_position, num_cls_token=self.num_cls_tokens)

                    pred_score = self.score_predictors[predictor_count](other_t + other_r, token_policy)
                    score = pred_score[:, :, 0]
                    num_keep_node = int(self.token_ratio[predictor_count] * M)
                    keep_policy = torch.topk(score, k=num_keep_node, dim=1, largest=True).indices
                    keep_policy = torch.sort(keep_policy, dim=1).values
                    other_t = batch_index_select(other_t, keep_policy)
                    other_r = batch_index_select(other_r, keep_policy)

                    interval = num_keep_node // (self.num_cls_tokens + 1)
                    cls_token_position = torch.arange(interval, interval * (self.num_cls_tokens + 1) + self.num_cls_tokens, interval + 1, device=cls_token_position.device).unsqueeze(0).expand(B, -1)
                    assert cls_token_position.shape == (B, self.num_cls_tokens), f'expect cls_token_position to be {(B, self.num_cls_tokens)} but got {cls_token_position.shape}'

                    hidden_states = combine_cls_and_other(cls_t, other_t, cls_token_position, num_cls_token=self.num_cls_tokens)
                    residual = combine_cls_and_other(cls_r, other_r, cls_token_position, num_cls_token=self.num_cls_tokens)
                    token_policy = torch.ones((B, num_keep_node, 1), dtype=hidden_states.dtype,   device=hidden_states.device)
                predictor_count += 1       

            hidden_states, residual, block_policy = layer(
                hidden_states, residual,
                inference_params=inference_params, 
                mask=mask if self.training else None, cls_token_position=cls_token_position)
            block_policies.append(block_policy)
        
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
       
        if self.cls_reduce > 1:
            if self.training:
                cls_t, other_t = extract_cls_and_other(hidden_states, cls_token_position, num_cls_token=self.num_cls_tokens)
                block_policy = torch.stack(block_policies, dim=0)
                return self.norm_neck(self.neck(cls_t).view(B, -1)), other_t, token_policy.detach(), pred_decisions, current_pos, block_policy
            else:
                cls_t, _ = extract_cls_and_other(hidden_states, cls_token_position, num_cls_token=self.num_cls_tokens)
                block_policy = torch.stack(block_policies, dim=0)
                return self.norm_neck(self.neck(cls_t).view(B, -1)), block_policy
        else:
            raise NotImplementedError

        if self.num_cls_tokens > 0:
            return hidden_states[:, self.cls_positions].view(B, -1)

        return hidden_states

    def forward(self, x, return_features=False, inference_params=None):
        if self.training:
            cls_t, other_t, token_policy, pred_decisions, current_pos, block_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), other_t, token_policy, pred_decisions, current_pos, block_policy
        else:
            cls_t, block_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), block_policy

class MambaRegTeacher(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16,
                 depth=24, 
                 embed_dim=192,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 num_cls_tokens=1,
                 cls_reduce=1,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.num_cls_tokens = num_cls_tokens
        self.cls_reduce = cls_reduce

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
            self.pos_embed_cls = nn.Parameter(
                torch.zeros(1, num_cls_tokens, self.embed_dim))
            H, W = self.patch_embed.grid_size
            self.token_idx, self.cls_positions = get_cls_idx(H, W, num_cls_tokens)
            
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if cls_reduce > 1:
            self.neck = nn.Linear(self.num_features, self.num_features // cls_reduce, bias=False)
            self.norm_neck = (nn.LayerNorm if not rms_norm else RMSNorm)(
                embed_dim * num_cls_tokens // cls_reduce, eps=norm_epsilon, **factory_kwargs)
            
        if num_classes < 1:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.num_features * (num_cls_tokens // cls_reduce), num_classes)

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_divide_out=True,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        if cls_reduce > 1:
            self.neck.apply(segm_init_weights)            
        if self.num_cls_tokens > 0:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.pos_embed_cls, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "pos_embed_cls"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token       
        x = self.patch_embed(x)
        B, _, _ = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.num_cls_tokens > 0:
            cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
            x = torch.cat([x, cls_token], dim=1)[:, self.token_idx]

        # mamba impl
        residual = None
        hidden_states = x
        for n, layer in enumerate(self.layers):          
            hidden_states, residual = layer(
                hidden_states, residual,
                inference_params=inference_params)
        
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
       
        if self.cls_reduce > 1:
            cls_token_position = self.cls_positions.unsqueeze(0).expand(B, -1)
            cls_t, other_t = extract_cls_and_other(hidden_states, cls_token_position, num_cls_token=self.num_cls_tokens)
            return self.norm_neck(self.neck(cls_t).view(B, -1)), other_t
        else:
            raise NotImplementedError

        if self.num_cls_tokens > 0:
            return hidden_states[:, self.cls_positions].view(B, -1)

        return hidden_states

    def forward(self, x, return_features=False, inference_params=None):
        x, other_t = self.forward_features(x, inference_params)
        return self.head(x), other_t

@register_model
def DyVMTiny(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def DyVMSmall(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, cls_reduce=2, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def DyVMBase(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, cls_reduce=4, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_base_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def DyVMLarge(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, embed_dim=1024, depth=48, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=16, cls_reduce=8, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_large_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def MambaRegTinyTeacher(pretrained=False, **kwargs):
    model = MambaRegTeacher(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def MambaRegSmallTeacher(pretrained=False, **kwargs):
    model = MambaRegTeacher(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, cls_reduce=2, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_small_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def MambaRegBaseTeacher(pretrained=False, **kwargs):
    model = MambaRegTeacher(
        patch_size=16, embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=12, cls_reduce=4, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_base_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def MambaRegLargeTeacher(pretrained=False, **kwargs):
    model = MambaRegTeacher(
        patch_size=16, embed_dim=1024, depth=48, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, num_cls_tokens=16, cls_reduce=8, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/Wangf3014/Mamba-Reg/resolve/main/mambar_large_patch16_224.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model