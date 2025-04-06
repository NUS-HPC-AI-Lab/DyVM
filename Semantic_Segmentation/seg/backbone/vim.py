import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

# add the root path to the system path
import sys, os
print(os.path.dirname(os.getcwd()) + "/vim")
# import the parent directory of the current cwd
sys.path.append(os.path.dirname(os.getcwd()) + "/vim")
from dyvm import DyVM, layer_norm_fn, rms_norm_fn, RMSNorm
from utils import interpolate_pos_embed


@BACKBONES.register_module()
class VisionMambaSeg(DyVM):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        num_classes=80,
        embed_dim=192,
        depth=24,
        d_state=16,
        use_checkpoint=False,
        pretrained=None,
        out_indices=[3, 5, 7, 11],
        if_fpn=True,
        use_residual_as_feature=False,
        last_layer_process="none",
        pruning_loc=[6, 12, 18],
        token_ratio=[0.7, 0.7**2, 0.7**3],
        **kwargs
    ):

        # for rope
        ft_seq_len = img_size // patch_size
        kwargs['ft_seq_len'] = ft_seq_len

        super().__init__(img_size, patch_size, stride, depth, embed_dim, d_state, in_chans, num_classes, pruning_loc=pruning_loc, token_ratio=token_ratio, **kwargs)

        self.use_checkpoint = use_checkpoint
        self.out_indices = out_indices
        self.if_fpn = if_fpn
        self.use_residual_as_feature = use_residual_as_feature
        self.last_layer_process = last_layer_process

        # del the parent class's head
        del self.head

        if if_fpn:
            if patch_size == 16:
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn2 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn3 = nn.Identity()

                self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            elif patch_size == 8:
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn2 = nn.Identity()

                self.fpn3 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fpn4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=4, stride=4),
                )

        self.init_weights(pretrained)

        for i in range(len(self.layers)):
            torch.nn.init.constant_(self.layers[i].mixer.block_head.weight, 0.)
            if self.layers[i].mixer.block_head.bias is not None:
                torch.nn.init.constant_(self.layers[i].mixer.block_head.bias, 5.0)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()

            # load_checkpoint(self, pretrained, strict=False, logger=logger)

            state_dict = torch.load(pretrained, map_location="cpu")
            # import ipdb; ipdb.set_trace()
            state_dict_model = state_dict["model"]
            state_dict_model.pop("head.weight")
            state_dict_model.pop("head.bias")
            # pop rope
            try:
                state_dict_model.pop("rope.freqs_cos")
                state_dict_model.pop("rope.freqs_sin")
            except:
                print("no rope in the pretrained model")

            if self.patch_embed.patch_size[-1] != state_dict["model"]["patch_embed.proj.weight"].shape[-1]:
                state_dict_model.pop("patch_embed.proj.weight")
                state_dict_model.pop("patch_embed.proj.bias")
            interpolate_pos_embed(self, state_dict_model)

            res = self.load_state_dict(state_dict_model, strict=False) 
            logger.info(res)
            print(res)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_plain_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        residual = None
        hidden_states = x
        features = []

        # todo: configure this
        # get two layers in a single for-loop
        for i, layer in enumerate(self.layers):

            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        return residual
    
    def forward_features(self, x, inference_params=None):
        assert self.use_residual_as_feature == False

        B, C, H, W = x.shape
        # x, (Hp, Wp) = self.patch_embed(x)
        x = self.patch_embed(x)

        batch_size, seq_len, _ = x.size()
        Hp = Wp = int(math.sqrt(seq_len))
        L = seq_len

        if self.pos_embed is not None:
            x = x + self.pos_embed
            x = self.pos_drop(x)

    
        hidden_states = x
        residual = None
        
        features = []
        predictor_count = 0
        token_policy = torch.ones((B, L, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        mask = torch.ones((B, L, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        if self.training:
            new_order = torch.sort(-(token_policy.squeeze(-1)), dim=1, stable=True)[1]
            reverse_order = torch.empty_like(new_order, dtype=torch.long)
            reverse_order = torch.sort(new_order, dim=1)[1]
        else:
            new_order = torch.sort(-(token_policy.squeeze(-1)), dim=1, stable=True)[1]
        pred_decisions = []
        block_policies = []

        feature_hidden_states = torch.zeros_like(hidden_states)
        feature_residual = torch.zeros_like(hidden_states)

        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):

                if self.training:
                    restored_hidden_states = torch.gather(hidden_states, 1, reverse_order.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
                    hidden_states = restored_hidden_states
                    if residual is not None:
                        restored_residual = torch.gather(residual, 1, reverse_order.unsqueeze(-1).expand(-1, -1, residual.size(-1)))
                        residual = restored_residual
                else:
                    restored_hidden_states = torch.zeros_like(x)
                    restored_hidden_states.scatter_(1, new_order.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)), hidden_states)
                    hidden_states = restored_hidden_states
                    if residual is not None:
                        restored_residual = torch.zeros_like(x)
                        restored_residual.scatter_(1, new_order.unsqueeze(-1).expand(-1, -1, residual.size(-1)), residual)
                        residual = restored_residual

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if i in self.pruning_loc or i-1 in self.out_indices:
                    if i-1 in self.out_indices:
                        features.append((hidden_states + feature_hidden_states).permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

                    if i in self.pruning_loc:
                        feature_hidden_states = hidden_states + feature_hidden_states
                        feature_residual = residual + feature_residual
                        if self.training:
                            pred_score = self.score_predictors[predictor_count](feature_hidden_states + feature_residual, token_policy)
                            hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * token_policy
                            pred_decisions.append(hard_keep_decision.squeeze(-1))
                            token_policy = hard_keep_decision

                            hidden_states = feature_hidden_states * token_policy
                            residual = feature_residual * token_policy
                            feature_hidden_states = feature_hidden_states * (1 - token_policy)
                            feature_residual = feature_residual * (1 - token_policy)

                            new_order = torch.sort(-(token_policy.squeeze(-1)), dim=1, stable=True)[1]
                            reverse_order = torch.empty_like(new_order, dtype=torch.long)
                            reverse_order = torch.sort(new_order, dim=1)[1]
                            rearranged_mask = torch.gather((token_policy.squeeze(-1)), 1, new_order).unsqueeze(-1)
                            mask = rearranged_mask
                        else:
                            pred_score = self.score_predictors[predictor_count](feature_hidden_states + feature_residual, token_policy)
                            score = pred_score[:, :, 0]
                            score = torch.where(token_policy.squeeze(-1) == 1, score, float('-inf'))
                            num_keep_node = int(self.token_ratio[predictor_count] * L)
                            keep_policy = torch.topk(score, k=num_keep_node, dim=1, largest=True).indices
                            keep_mask = torch.zeros_like(token_policy)
                            batch_indices = torch.arange(B, device=token_policy.device).unsqueeze(-1).expand(-1, num_keep_node)
                            keep_mask[batch_indices, keep_policy] = 1
                            token_policy = keep_mask

                            hidden_states = feature_hidden_states * token_policy
                            residual = feature_residual * token_policy
                            feature_hidden_states = feature_hidden_states * (1 - token_policy)
                            feature_residual = feature_residual * (1 - token_policy)

                            new_order = torch.sort(-(token_policy.squeeze(-1)), dim=1, stable=True)[1]
                            new_order = new_order[:, :num_keep_node]
                    
                        predictor_count += 1

                rearranged_hidden_states = torch.gather(hidden_states, 1, new_order.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
                hidden_states = rearranged_hidden_states
                if residual is not None:
                    rearranged_residual = torch.gather(residual, 1, new_order.unsqueeze(-1).expand(-1, -1, residual.size(-1)))
                    residual = rearranged_residual

                hidden_states, residual, block_policy = layer(
                    hidden_states, residual, inference_params=inference_params, mask = mask if self.training else None
                )
                block_policies.append(block_policy)
        else:
            raise NotImplementedError

        if self.training:
            restored_hidden_states = torch.gather(hidden_states, 1, reverse_order.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
            restored_residual = torch.gather(residual, 1, reverse_order.unsqueeze(-1).expand(-1, -1, residual.size(-1)))
            feature_hidden_states = restored_hidden_states + feature_hidden_states
            feature_residual = restored_residual + feature_residual
        else:
            restored_hidden_states = torch.zeros_like(x)
            restored_residual = torch.zeros_like(x)
            restored_hidden_states.scatter_(1, new_order.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)), hidden_states)
            restored_residual.scatter_(1, new_order.unsqueeze(-1).expand(-1, -1, residual.size(-1)), residual)
            feature_hidden_states = restored_hidden_states + feature_hidden_states
            feature_residual = restored_residual + feature_residual

        if 23 in self.out_indices:
            features.append(feature_hidden_states.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

        # residual = None
        # hidden_states = x
        # features = []
        # if not self.if_bidirectional:
        #     for i, layer in enumerate(self.layers):

        #         # rope about
        #         if self.if_rope:
        #             hidden_states = self.rope(hidden_states)
        #             if residual is not None and self.if_rope_residual:
        #                 residual = self.rope(residual)

        #         hidden_states, residual = layer(
        #             hidden_states, residual, inference_params=inference_params
        #         )

                # if self.use_residual_as_feature:
                #     if i-1 in self.out_indices:
                #         # residual_p = residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                #         # features.append(residual_p.contiguous())
                #         features.append(residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())
                # else:
                #     if i in self.out_indices:
                #         # hidden_states_p = hidden_states.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                #         # features.append(hidden_states_p.contiguous()) 
                #         features.append(hidden_states.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

        # else:
        #     raise NotImplementedError

        if self.last_layer_process == 'none':
            residual = hidden_states
        elif self.last_layer_process == 'add':
            residual = hidden_states + residual
        elif self.last_layer_process == 'add & norm':
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            residual = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # if self.use_residual_as_feature and self.out_indices[-1] == len(self.layers)-1:
        #     # residual_p = residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        #     # features.append(residual_p.contiguous())
        #     features.append(residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

        if self.if_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            assert len(features) == len(ops)

            if len(features) == 1:
                for i in range(len(ops) - 1):
                    features.append(features[0])
                for i in range(len(features)):
                    features[i] = ops[i](features[i])
            else:
                for i in range(len(features)):
                    features[i] = ops[i](features[i])

        if self.training:
            token_ratio_loss = 0.0
            token_ratio_loss_weight = 20.0
            target_token_ratio = self.token_ratio
            for i, score in enumerate(pred_decisions):
                pos_ratio = score.mean(1)
                token_ratio_loss += ((pos_ratio - target_token_ratio[i]) ** 2).mean() / len(target_token_ratio)

            token_ratio_loss = token_ratio_loss * token_ratio_loss_weight

            target_block_ratio = 0.8
            block_ratio_loss_weight = 20.0
            block_policy = torch.stack(block_policies, dim=0)
            block_ratio_loss = ((block_policy.mean() - target_block_ratio) ** 2)

            block_ratio_loss = block_ratio_loss * block_ratio_loss_weight
            return tuple(features), token_ratio_loss, block_ratio_loss
        else:
            return tuple(features)

    def forward(self, x):
        if self.training:
            x, token_ratio_loss, block_ratio_loss = self.forward_features(x)
            return x, token_ratio_loss, block_ratio_loss
        else:
            x = self.forward_features(x)
            return x