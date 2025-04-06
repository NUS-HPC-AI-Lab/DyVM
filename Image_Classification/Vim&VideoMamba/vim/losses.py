# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


# class DyVMLoss_Token_Block_Layer(torch.nn.Module):
#     """
#     This module wraps a standard criterion and adds an extra knowledge distillation loss by
#     taking a teacher model prediction and using it as additional supervision.
#     """
#     def __init__(self, 
#                  teacher_model, 
#                  base_criterion: torch.nn.Module, 
#                  token_pruning_weight=2.0, 
#                  block_pruning_weight=2.0, 
#                  layer_pruning_weight=2.0, 
#                  distill_weight=0.5, 
#                  dynamic=False, 
#                  pruning_loc=[3,6,9], 
#                  token_keep_ratio=[0.75, 0.5, 0.25],
#                  block_keep_ratio=0.8, 
#                  layer_keep_ratio=0.8, 
#                  clf_weight=1.0, 
#                  mse_token=False, 
#                  print_mode=True):
#         super().__init__()
#         self.teacher_model = teacher_model
#         self.base_criterion = base_criterion
#         self.clf_weight = clf_weight
#         self.pruning_loc = pruning_loc
#         self.token_keep_ratio = token_keep_ratio
#         self.block_keep_ratio = block_keep_ratio
#         self.layer_keep_ratio = layer_keep_ratio
#         self.count = 0
#         self.print_mode = print_mode
#         self.cls_loss = 0
#         self.ratio_loss = 0
#         self.cls_distill_loss = 0
#         self.token_distill_loss = 0
#         self.mse_token = mse_token
#         self.dynamic = dynamic
        
#         self.token_pruning_weight = token_pruning_weight
#         self.block_pruning_weight = block_pruning_weight
#         self.layer_pruning_weight = layer_pruning_weight
#         self.distill_weight = distill_weight

#         print('token_pruning_weight: ', token_pruning_weight, 
#               'block_pruning_weight: ', block_pruning_weight, 
#               'layer_pruning_weight: ', layer_pruning_weight, 
#               'distill_weight: ', distill_weight)

#         if dynamic:
#             print('using dynamic loss')

#     def forward(self, inputs, outputs, labels):
#         """
#         Args:
#             inputs: The original inputs that are feed to the teacher model
#             outputs: the outputs of the model to be trained. It is expected to be
#                 either a Tensor, or a Tuple[Tensor, Tensor], with the original output
#                 in the first position and the distillation predictions as the second output
#             labels: the labels for the base criterion
#         """

#         cls_t, other_t, token_policy, pred_decisions, current_pos, block_policy, layer_policy = outputs

#         # classification loss (prediction and true label)
#         cls_loss = self.base_criterion(cls_t, labels)

#         # token ratio loss
#         target_token_ratio = self.token_keep_ratio
#         token_ratio_loss = 0.0
#         for i, score in enumerate(pred_decisions):
#             if self.dynamic:
#                 pos_ratio = score.mean()
#             else:
#                 pos_ratio = score.mean(1)
#             token_ratio_loss += ((pos_ratio - target_token_ratio[i]) ** 2).mean()
        
#         # block ratio loss
#         target_block_ratio = self.block_keep_ratio
#         block_ratio_loss = ((block_policy.mean() - target_block_ratio) ** 2)

#         # layer ratio loss
#         target_layer_ratio = self.layer_keep_ratio
#         layer_ratio_loss = ((layer_policy.mean() - target_layer_ratio) ** 2)

#         # KL loss
#         with torch.no_grad():
#             teacher_cls_t, teacher_other_t = self.teacher_model(inputs)

#         KL_loss = F.kl_div(
#             F.log_softmax(cls_t, dim=1),
#             F.log_softmax(teacher_cls_t, dim=1),
#             reduction='batchmean',
#             log_target=True
#         )

#         # distillation loss
#         B, L, D = other_t.shape
#         teacher_other_t = teacher_other_t[torch.arange(B).unsqueeze(1), current_pos]
#         bool_token_policy = token_policy.reshape(B * L) > 0.5
#         if bool_token_policy.sum() < 0.1:
#             token_distill_loss = 0
#         else:
#             other_t = other_t.reshape(B * L, D)[bool_token_policy]
#             teacher_other_t = teacher_other_t.reshape(B * L, D)[bool_token_policy]
#             if self.mse_token:
#                 token_distill_loss = torch.pow(other_t - teacher_other_t, 2).mean()
#             else:
#                 token_distill_loss = F.kl_div(
#                     F.log_softmax(other_t, dim=1),
#                     F.log_softmax(teacher_other_t, dim=1),
#                     reduction='batchmean',
#                     log_target=True
#                 )

#         loss = self.clf_weight * cls_loss + self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc) + self.block_pruning_weight * block_ratio_loss + self.layer_pruning_weight * layer_ratio_loss + self.distill_weight * (KL_loss + token_distill_loss)

#         return loss, [self.clf_weight * cls_loss, 
#                       self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc), 
#                       self.block_pruning_weight * block_ratio_loss, 
#                       self.layer_pruning_weight * layer_ratio_loss, 
#                       self.distill_weight * KL_loss, 
#                       self.distill_weight * token_distill_loss]


# class DyVMLoss_Token_Layer(torch.nn.Module):
#     """
#     This module wraps a standard criterion and adds an extra knowledge distillation loss by
#     taking a teacher model prediction and using it as additional supervision.
#     """
#     def __init__(self, 
#                  teacher_model, 
#                  base_criterion: torch.nn.Module,
#                  pruning_loc=[3,6,9],
#                  token_pruning_weight=2.0,
#                  layer_pruning_weight=2.0, 
#                  distill_weight=0.5, 
#                  dynamic=False, 
#                  token_keep_ratio=[0.75, 0.5, 0.25],
#                  layer_keep_ratio=0.8, 
#                  clf_weight=1.0, 
#                  mse_token=False, 
#                  print_mode=True):
#         super().__init__()
#         self.teacher_model = teacher_model
#         self.base_criterion = base_criterion
#         self.clf_weight = clf_weight
#         self.pruning_loc = pruning_loc
#         self.token_keep_ratio = token_keep_ratio
#         self.layer_keep_ratio = layer_keep_ratio
#         self.count = 0
#         self.print_mode = print_mode
#         self.cls_loss = 0
#         self.ratio_loss = 0
#         self.cls_distill_loss = 0
#         self.token_distill_loss = 0
#         self.mse_token = mse_token
#         self.dynamic = dynamic
        
#         self.token_pruning_weight = token_pruning_weight
#         self.layer_pruning_weight = layer_pruning_weight
#         self.distill_weight = distill_weight

#         print('token_pruning_weight: ', token_pruning_weight, 
#               'layer_pruning_weight: ', layer_pruning_weight, 
#               'distill_weight: ', distill_weight)

#         if dynamic:
#             print('using dynamic loss')

#     def forward(self, inputs, outputs, labels):
#         """
#         Args:
#             inputs: The original inputs that are feed to the teacher model
#             outputs: the outputs of the model to be trained. It is expected to be
#                 either a Tensor, or a Tuple[Tensor, Tensor], with the original output
#                 in the first position and the distillation predictions as the second output
#             labels: the labels for the base criterion
#         """

#         cls_t, other_t, token_policy, pred_decisions, current_pos, layer_policy = outputs

#         # classification loss (prediction and true label)
#         cls_loss = self.base_criterion(cls_t, labels)

#         # token ratio loss
#         target_token_ratio = self.token_keep_ratio
#         token_ratio_loss = 0.0
#         for i, score in enumerate(pred_decisions):
#             if self.dynamic:
#                 pos_ratio = score.mean()
#             else:
#                 pos_ratio = score.mean(1)
#             token_ratio_loss += ((pos_ratio - target_token_ratio[i]) ** 2).mean()

#         # layer ratio loss
#         target_layer_ratio = self.layer_keep_ratio
#         layer_ratio_loss = ((layer_policy.mean() - target_layer_ratio) ** 2)

#         # KL loss
#         with torch.no_grad():
#             teacher_cls_t, teacher_other_t = self.teacher_model(inputs)

#         KL_loss = F.kl_div(
#             F.log_softmax(cls_t, dim=1),
#             F.log_softmax(teacher_cls_t, dim=1),
#             reduction='batchmean',
#             log_target=True
#         )

#         # distillation loss
#         B, L, D = other_t.shape
#         teacher_other_t = teacher_other_t[torch.arange(B).unsqueeze(1), current_pos]
#         bool_token_policy = token_policy.reshape(B * L) > 0.5
#         if bool_token_policy.sum() < 0.1:
#             token_distill_loss = 0
#         else:
#             other_t = other_t.reshape(B * L, D)[bool_token_policy]
#             teacher_other_t = teacher_other_t.reshape(B * L, D)[bool_token_policy]
#             if self.mse_token:
#                 token_distill_loss = torch.pow(other_t - teacher_other_t, 2).mean()
#             else:
#                 token_distill_loss = F.kl_div(
#                     F.log_softmax(other_t, dim=1),
#                     F.log_softmax(teacher_other_t, dim=1),
#                     reduction='batchmean',
#                     log_target=True
#                 )

#         loss = self.clf_weight * cls_loss + \
#                self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc) + \
#                self.layer_pruning_weight * layer_ratio_loss + \
#                self.distill_weight * (KL_loss + token_distill_loss)

#         return loss, [self.clf_weight * cls_loss, 
#                       self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc), 
#                       self.layer_pruning_weight * layer_ratio_loss, 
#                       self.distill_weight * KL_loss, 
#                       self.distill_weight * token_distill_loss]


class DyVMLoss_Token_Block(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, 
                 teacher_model, 
                 base_criterion: torch.nn.Module, 
                 token_pruning_weight=2.0, 
                 block_pruning_weight=2.0, 
                 distill_weight=0.5, 
                 dynamic=False, 
                 pruning_loc=[3,6,9], 
                 token_keep_ratio=[0.75, 0.5, 0.25],
                 block_keep_ratio=0.8, 
                 clf_weight=1.0, 
                 mse_token=False, 
                 print_mode=True,
                 **kwargs):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.token_keep_ratio = token_keep_ratio
        self.block_keep_ratio = block_keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic
        
        self.token_pruning_weight = token_pruning_weight
        self.block_pruning_weight = block_pruning_weight
        self.distill_weight = distill_weight

        print('token_pruning_weight: ', token_pruning_weight, 
              'block_pruning_weight: ', block_pruning_weight, 
              'distill_weight: ', distill_weight)

        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        cls_t, other_t, token_policy, pred_decisions, current_pos, block_policy = outputs

        # classification loss (prediction and true label)
        cls_loss = self.base_criterion(cls_t, labels)

        # token ratio loss
        target_token_ratio = self.token_keep_ratio
        token_ratio_loss = 0.0
        for i, score in enumerate(pred_decisions):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            token_ratio_loss += ((pos_ratio - target_token_ratio[i]) ** 2).mean()
        
        # block ratio loss
        target_block_ratio = self.block_keep_ratio
        block_ratio_loss = ((block_policy.mean() - target_block_ratio) ** 2)

        # KL loss
        with torch.no_grad():
            teacher_cls_t, teacher_other_t = self.teacher_model(inputs)

        KL_loss = F.kl_div(
            F.log_softmax(cls_t, dim=1),
            F.log_softmax(teacher_cls_t, dim=1),
            reduction='batchmean',
            log_target=True
        )

        # distillation loss
        B, L, D = other_t.shape
        teacher_other_t = teacher_other_t[torch.arange(B).unsqueeze(1), current_pos]
        bool_token_policy = token_policy.reshape(B * L) > 0.5
        if bool_token_policy.sum() < 0.1:
            token_distill_loss = 0
        else:
            other_t = other_t.reshape(B * L, D)[bool_token_policy]
            teacher_other_t = teacher_other_t.reshape(B * L, D)[bool_token_policy]
            if self.mse_token:
                token_distill_loss = torch.pow(other_t - teacher_other_t, 2).mean()
            else:
                token_distill_loss = F.kl_div(
                    F.log_softmax(other_t, dim=1),
                    F.log_softmax(teacher_other_t, dim=1),
                    reduction='batchmean',
                    log_target=True
                )

        loss = self.clf_weight * cls_loss + \
               self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc) + \
               self.block_pruning_weight * block_ratio_loss + \
               self.distill_weight * (KL_loss + token_distill_loss)

        return loss, [self.clf_weight * cls_loss, 
                      self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc), 
                      self.block_pruning_weight * block_ratio_loss, 
                      self.distill_weight * KL_loss, 
                      self.distill_weight * token_distill_loss]


class DyVMLoss_Block_Only(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, 
                 teacher_model, 
                 base_criterion: torch.nn.Module, 
                 distill_weight=0.5, 
                 dynamic=False, 
                 block_keep_ratio=0.8, 
                 block_pruning_weight=1.0, 
                 clf_weight=1.0, 
                 mse_token=False, 
                 print_mode=True,
                 **kwargs):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.block_keep_ratio = block_keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic
        
        self.block_pruning_weight = block_pruning_weight
        self.distill_weight = distill_weight

        print('block_pruning_weight: ', block_pruning_weight, 'distill_weight: ', distill_weight)

        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        cls_t, block_policy = outputs

        # classification loss (prediction and true label)
        cls_loss = self.base_criterion(cls_t, labels)

        # block ratio loss
        target_block_ratio = self.block_keep_ratio
        block_ratio_loss = ((block_policy.mean() - target_block_ratio) ** 2)

        # KL loss
        with torch.no_grad():
            teacher_cls_t, _ = self.teacher_model(inputs)

        KL_loss = F.kl_div(
            F.log_softmax(cls_t, dim=1),
            F.log_softmax(teacher_cls_t, dim=1),
            reduction='batchmean',
            log_target=True
        )
        loss = self.clf_weight * cls_loss + self.block_pruning_weight * block_ratio_loss + self.distill_weight * KL_loss  

        return loss, [self.clf_weight * cls_loss, self.block_pruning_weight * block_ratio_loss, self.distill_weight * KL_loss]


class DyVMLoss_Token_Only(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, 
                 teacher_model, 
                 base_criterion: torch.nn.Module, 
                 token_pruning_weight=2.0, 
                 distill_weight=0.5, 
                 dynamic=False, 
                 pruning_loc=[3,6,9], 
                 keep_ratio=[0.75, 0.5, 0.25], 
                 clf_weight=1.0, 
                 mse_token=False, 
                 print_mode=True,
                 **kwargs):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic
        
        self.token_pruning_weight = token_pruning_weight
        self.distill_weight = distill_weight

        print('token_pruning_weight: ', token_pruning_weight, 'distill_weight: ', distill_weight)

        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        cls_t, other_t, policy, pred_decisions, current_pos = outputs

        # classification loss (prediction and true label)
        cls_loss = self.base_criterion(cls_t, labels)

        # token ratio loss
        target_ratio = self.keep_ratio
        token_ratio_loss = 0.0
        for i, score in enumerate(pred_decisions):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            token_ratio_loss += ((pos_ratio - target_ratio[i]) ** 2).mean()

        # KL loss
        with torch.no_grad():
            teacher_cls_t, teacher_other_t = self.teacher_model(inputs)

        KL_loss = F.kl_div(
            F.log_softmax(cls_t, dim=1),
            F.log_softmax(teacher_cls_t, dim=1),
            reduction='batchmean',
            log_target=True
        )

        # distillation loss
        B, L, D = other_t.shape
        teacher_other_t = teacher_other_t[torch.arange(B).unsqueeze(1), current_pos]
        bool_policy = policy.reshape(B * L) > 0.5
        if bool_policy.sum() < 0.1:
            token_distill_loss = 0
        else:
            other_t = other_t.reshape(B * L, D)[bool_policy]
            teacher_other_t = teacher_other_t.reshape(B * L, D)[bool_policy]
            if self.mse_token:
                token_distill_loss = torch.pow(other_t - teacher_other_t, 2).mean()
            else:
                token_distill_loss = F.kl_div(
                    F.log_softmax(other_t, dim=1),
                    F.log_softmax(teacher_other_t, dim=1),
                    reduction='batchmean',
                    log_target=True
                )

        loss = self.clf_weight * cls_loss + self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc) + self.distill_weight * (KL_loss + token_distill_loss)  

        return loss, [self.clf_weight * cls_loss, self.token_pruning_weight * token_ratio_loss / len(self.pruning_loc), self.distill_weight * KL_loss, self.distill_weight * token_distill_loss]


# class DyVMLoss_Layer_Only(torch.nn.Module):
#     """
#     This module wraps a standard criterion and adds an extra knowledge distillation loss by
#     taking a teacher model prediction and using it as additional supervision.
#     """
#     def __init__(self, 
#                  teacher_model, 
#                  base_criterion: torch.nn.Module, 
#                  distill_weight=0.5, 
#                  dynamic=False, 
#                  layer_keep_ratio=0.8, 
#                  layer_pruning_weight=1.0, 
#                  clf_weight=1.0, 
#                  mse_token=False, 
#                  print_mode=True):
#         super().__init__()
#         self.teacher_model = teacher_model
#         self.base_criterion = base_criterion
#         self.clf_weight = clf_weight
#         self.layer_keep_ratio = layer_keep_ratio
#         self.count = 0
#         self.print_mode = print_mode
#         self.cls_loss = 0
#         self.mse_token = mse_token
#         self.dynamic = dynamic
        
#         self.layer_pruning_weight = layer_pruning_weight
#         self.distill_weight = distill_weight

#         print('layer_pruning_weight: ', layer_pruning_weight, 'distill_weight: ', distill_weight)

#         if dynamic:
#             print('using dynamic loss')

#     def forward(self, inputs, outputs, labels):
#         """
#         Args:
#             inputs: The original inputs that are feed to the teacher model
#             outputs: the outputs of the model to be trained. It is expected to be
#                 either a Tensor, or a Tuple[Tensor, Tensor], with the original output
#                 in the first position and the distillation predictions as the second output
#             labels: the labels for the base criterion
#         """

#         cls_t, layer_policy = outputs

#         # classification loss (prediction and true label)
#         cls_loss = self.base_criterion(cls_t, labels)

#         # layer ratio loss
#         target_layer_ratio = self.layer_keep_ratio
#         layer_ratio_loss = ((layer_policy.mean() - target_layer_ratio) ** 2)

#         # KL loss
#         with torch.no_grad():
#             teacher_cls_t = self.teacher_model(inputs)

#         KL_loss = F.kl_div(
#             F.log_softmax(cls_t, dim=1),
#             F.log_softmax(teacher_cls_t, dim=1),
#             reduction='batchmean',
#             log_target=True
#         )
#         loss = self.clf_weight * cls_loss + self.layer_pruning_weight * layer_ratio_loss + self.distill_weight * KL_loss  

#         return loss, [self.clf_weight * cls_loss, self.layer_pruning_weight * layer_ratio_loss, self.distill_weight * KL_loss]
