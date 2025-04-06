import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.mamba_block import create_block_multi, create_block_multi_dynamic
from models.predictor import PredictorLG
from models.embedding import PatchEmbed

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights

from rope import *

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from utils import _init_weights, segm_init_weights, resize_pos_embed

import math

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
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 use_middle_cls_token=False,
                 directions=None,
                 pruning_loc=None, 
                 token_ratio=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.patch_size = patch_size
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim, strict_img_size=False, dynamic_img_pad=True)
        num_patches = self.patch_embed.num_patches
        self.token_size = self.patch_embed.grid_size

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            if isinstance(img_size, (tuple, list)):
                hw_seq_len = img_size[0] // patch_size
            else:
                hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        if directions is None:
            directions = [None] * depth
        self.layers = nn.ModuleList(
            [
                create_block_multi_dynamic(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    directions=directions[i],
                    use_middle_cls_token=use_middle_cls_token,
                    token_size=self.token_size,
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

        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, out_indices=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        M = x.shape[1]
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.use_middle_cls_token:
                token_position = x.shape[1] // 2
                # add cls token to the middle of sequence
                x = torch.cat([x[:, :token_position, :], cls_token, x[:, token_position:, :]], dim=1)
            else:
                # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            H, W = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
            for layer in self.layers:
                layer.mixer.multi_scan.token_size = (H, W)
            if H != self.token_size[0] or W != self.token_size[1]:
                # downstream tasks such as det and seg may have various input resolutions
                pos_embed = resize_pos_embed(self.pos_embed, (H, W), self.token_size, 'bicubic')
                if self.if_rope:
                    freqs_cos = resize_pos_embed(self.rope.freqs_cos.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
                    freqs_sin = resize_pos_embed(self.rope.freqs_sin.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
            else:
                pos_embed = self.pos_embed
                freqs_cos = None
                freqs_sin = None
            x = x + pos_embed
            x = self.pos_drop(x)

        outs = []

        # mamba impl
        residual = None
        hidden_states = x
        predictor_count = 0
        pred_decisions = []
        current_pos = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)
        token_policy = torch.ones((B, M, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        block_policies = []

        for layer_idx, layer in enumerate(self.layers):
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

            if layer_idx in self.pruning_loc:
                if self.training:
                    pred_score = self.score_predictors[predictor_count](hidden_states + residual, token_policy)
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * token_policy
                    pred_decisions.append(hard_keep_decision.squeeze(-1))
                    token_policy = hard_keep_decision
                else:
                    pred_score = self.score_predictors[predictor_count](hidden_states + residual, token_policy)
                    score = pred_score[:, :, 0]
                    score = torch.where(token_policy.squeeze(-1) == 1, score, float('-inf'))
                    num_keep_node = int(self.token_ratio[predictor_count] * M)
                    keep_policy = torch.topk(score, k=num_keep_node, dim=1, largest=True).indices
                    
                    keep_mask = torch.zeros_like(token_policy)
                    batch_indices = torch.arange(B, device=token_policy.device).unsqueeze(-1).expand(-1, num_keep_node)
                    keep_mask[batch_indices, keep_policy] = 1
                    token_policy = keep_mask
                    
                predictor_count += 1

            hidden_states, residual, block_policy = layer(
                hidden_states, residual, inference_params=inference_params, mask=token_policy
            )
            block_policies.append(block_policy)
            if out_indices is not None and layer_idx in out_indices:
                outs.append(hidden_states)

        if out_indices is not None:
            assert len(outs) == len(out_indices)
            return outs, (H, W)

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

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_middle_cls_token:
                return hidden_states[:, token_position, :]
            else:
                return hidden_states[:, 0, :]

        # if self.final_pool_type == 'none':
        #     return hidden_states[:, -1, :]
        # elif self.final_pool_type == 'mean':
        #     return hidden_states.mean(dim=1)
        # elif self.final_pool_type == 'max':
        #     return hidden_states.max(dim=1)
        # elif self.final_pool_type == 'all':
        #     return hidden_states
        # else:
        #     raise NotImplementedError
        block_policy = torch.stack(block_policies, dim=0)
        if self.training:
            return hidden_states.mean(dim=1), hidden_states, token_policy.detach(), pred_decisions, current_pos, block_policy
        else:
            return hidden_states.mean(dim=1), block_policy

    def forward(self, x, return_features=False, inference_params=None):
        if self.training:
            cls_t, other_t, token_policy, pred_decisions, current_pos, block_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), other_t, token_policy, pred_decisions, current_pos, block_policy
        else:
            cls_t, block_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), block_policy

class LocalVimTeacher(nn.Module):
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
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 use_middle_cls_token=False,
                 directions=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.patch_size = patch_size

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim, strict_img_size=False, dynamic_img_pad=True)
        num_patches = self.patch_embed.num_patches
        self.token_size = self.patch_embed.grid_size

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            if isinstance(img_size, (tuple, list)):
                hw_seq_len = img_size[0] // patch_size
            else:
                hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        if directions is None:
            directions = [None] * depth
        self.layers = nn.ModuleList(
            [
                create_block_multi(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    directions=directions[i],
                    use_middle_cls_token=use_middle_cls_token,
                    token_size=self.token_size,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, out_indices=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.use_middle_cls_token:
                token_position = x.shape[1] // 2
                # add cls token to the middle of sequence
                x = torch.cat([x[:, :token_position, :], cls_token, x[:, token_position:, :]], dim=1)
            else:
                # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            H, W = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
            for layer in self.layers:
                layer.mixer.multi_scan.token_size = (H, W)
            if H != self.token_size[0] or W != self.token_size[1]:
                # downstream tasks such as det and seg may have various input resolutions
                pos_embed = resize_pos_embed(self.pos_embed, (H, W), self.token_size, 'bicubic')
                if self.if_rope:
                    freqs_cos = resize_pos_embed(self.rope.freqs_cos.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
                    freqs_sin = resize_pos_embed(self.rope.freqs_sin.unsqueeze(0), (H, W), self.token_size, 'bicubic')[0]
            else:
                pos_embed = self.pos_embed
                freqs_cos = None
                freqs_sin = None
            x = x + pos_embed
            x = self.pos_drop(x)

        outs = []

        # mamba impl
        residual = None
        hidden_states = x
        for layer_idx, layer in enumerate(self.layers):
            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

            if out_indices is not None and layer_idx in out_indices:
                outs.append(hidden_states)

        if out_indices is not None:
            assert len(outs) == len(out_indices)
            return outs, (H, W)

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

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_middle_cls_token:
                return hidden_states[:, token_position, :]
            else:
                return hidden_states[:, 0, :]

        # if self.final_pool_type == 'none':
        #     return hidden_states[:, -1, :]
        # elif self.final_pool_type == 'mean':
        #     return hidden_states.mean(dim=1)
        # elif self.final_pool_type == 'max':
        #     return hidden_states.max(dim=1)
        # elif self.final_pool_type == 'all':
        #     return hidden_states
        # else:
        #     raise NotImplementedError
        return hidden_states.mean(dim=1), hidden_states

    def forward(self, x, return_features=False, inference_params=None):
        x, hidden_states = self.forward_features(x, inference_params)
        x = self.head(x)
        return x, hidden_states

@register_model
def DyVMTiny(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = DyVM(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions,
        if_cls_token=False, use_middle_cls_token=False, **kwargs)
    # if_cls_token=True, if_devide_out=True, use_middle_cls_token=True
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
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = DyVM(
        patch_size=16, embed_dim=384, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def LocalMambaTinyTeacher(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = LocalVimTeacher(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions,
        if_cls_token=False, use_middle_cls_token=False, **kwargs)
    # if_cls_token=True, if_devide_out=True, use_middle_cls_token=True
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def LocalMambaSmallTeacher(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = LocalVimTeacher(
        patch_size=16, embed_dim=384, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model