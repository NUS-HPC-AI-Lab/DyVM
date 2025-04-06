# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.mamba_block import DynamicBlock, Block
from mamba_ssm.modules.mamba_simple import DynamicMamba, Mamba
from models.predictor import PredictorLG
from models.embedding import PatchEmbed

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from utils import rearrange_pos, batch_index_select, _init_weights, segm_init_weights

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type='v2', if_divide_out=False, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def create_block_dynamic(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba=True,
    block_keep_ratio=None
):

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(DynamicMamba, d_state=d_state, layer_idx=layer_idx, bimamba_type="v2", 
                        if_divide_out=False, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = DynamicBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        block_keep_ratio=block_keep_ratio,
    )
    block.layer_idx = layer_idx
    return block


class DyVM(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            stride=16,
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
            pruning_loc=None, 
            token_ratio=None,
            block_keep_ratio=None,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
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
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    block_keep_ratio=block_keep_ratio,
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
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

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
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        B, M, _ = x.shape
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        if self.training:
            cls_token_position = torch.full((B,), 0, dtype=torch.long, device=x.device)
        else:
            cls_token_position = 0

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        predictor_count = 0
        pred_decisions = []
        current_pos = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)
        policy = torch.ones((B, M, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        mask = torch.ones((B, M + 1, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        path_policies = []
        for i, layer in enumerate(self.layers):
            if i in self.pruning_loc:
                if self.training:
                    cls_t = hidden_states[:, 0:1, :]
                    other_t = hidden_states[:, 1:, :]
                    cls_r = residual[:, 0:1, :]
                    other_r = residual[:, 1:, :]
                    pred_score = self.score_predictors[predictor_count](other_t + other_r, policy)
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * policy
                    pred_decisions.append(hard_keep_decision.squeeze(-1))
                    hard_keep_decision, other_t, other_r, current_pos, _ = rearrange_pos(hard_keep_decision, other_t, other_r, current_pos)
                    hidden_states = torch.cat([cls_t, other_t], dim=1)
                    residual = torch.cat([cls_r, other_r], dim=1)
                    mask = torch.cat([torch.ones((B, 1, 1), dtype=hidden_states.dtype, device=hidden_states.device), hard_keep_decision], dim=1)
                    policy = hard_keep_decision
                else:
                    cls_t = hidden_states[:, 0:1, :]
                    other_t = hidden_states[:, 1:, :]
                    cls_r = residual[:, 0:1, :]
                    other_r = residual[:, 1:, :]
                    pred_score = self.score_predictors[predictor_count](other_t + other_r, policy)
                    score = pred_score[:, :, 0]
                    num_keep_node = int(self.token_ratio[predictor_count] * M)
                    keep_policy = torch.topk(score, k=num_keep_node, dim=1, largest=True).indices
                    keep_policy = torch.sort(keep_policy, dim=1).values
                    other_t = batch_index_select(other_t, keep_policy)
                    other_r = batch_index_select(other_r, keep_policy)
                    hidden_states = torch.cat([cls_t, other_t], dim=1)
                    residual = torch.cat([cls_r, other_r], dim=1)
                    policy = torch.ones((B, num_keep_node, 1), dtype=hidden_states.dtype,   device=hidden_states.device)
                predictor_count += 1


            hidden_states, residual, path_policy = layer(
                hidden_states, residual, inference_params=inference_params, 
                mask= mask if self.training else None, cls_token_position=cls_token_position
            )
            path_policies.append(path_policy)

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

        if self.training:
            cls_t = hidden_states[:, 0, :]
            other_t = hidden_states[:, 1:, :]
            path_policy = torch.stack(path_policies, dim=0)
            return cls_t, other_t, policy.detach(), pred_decisions, current_pos, path_policy
        else:
            cls_t = hidden_states[:, 0, :]
            path_policy = torch.stack(path_policies, dim=0)
            return cls_t, path_policy

    def forward(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        if self.training:
            cls_t, other_t, policy, pred_decisions, current_pos, path_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), other_t, policy, pred_decisions, current_pos, path_policy
        else:
            cls_t, path_policy = self.forward_features(x, inference_params)
            return self.head(cls_t), path_policy


class VideoMambaTeacher(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            stride=16,
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
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
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

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
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

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

        cls_token = hidden_states[:, 0, :]
        other_tokens = hidden_states[:, 1:, :]
        return cls_token, other_tokens

    def forward(self, x, inference_params=None):
        cls_token, other_tokens = self.forward_features(x, inference_params)
        cls_pred = self.head(cls_token)
        return cls_pred, other_tokens


@register_model
def DyVMTiny(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def VideoMambaTinyTeacher(pretrained=False, **kwargs):
    model = VideoMambaTeacher(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def DyVMSmall(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, 
        embed_dim=384,
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def VideoMambaSmallTeacher(pretrained=False, **kwargs):
    model = VideoMambaTeacher(
        patch_size=16, 
        embed_dim=384,
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def DyVMMiddle(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, 
        embed_dim=576,
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def VideoMambaMiddleTeacher(pretrained=False, **kwargs):
    model = VideoMambaTeacher(
        patch_size=16, 
        embed_dim=576,
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def DyVMBase(pretrained=False, **kwargs):
    model = DyVM(
        patch_size=16, 
        embed_dim=768,
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def VideoMambaBaseTeacher(pretrained=False, **kwargs):
    model = VideoMambaTeacher(
        patch_size=16, 
        embed_dim=768,
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model