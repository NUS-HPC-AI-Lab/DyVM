# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, dynamic=False, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], clf_weight=1.0, mse_token=False, print_mode=True):
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

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        print('ratio_weight: ', ratio_weight, 'distill_weight: ', distill_weight)


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

        cls_t, other_t, policy, current_pos = outputs

        # classification loss (prediction and true label)
        cls_loss = self.base_criterion(cls_t, labels)

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
            token_distill_loss = other_t.new(1,).fill_(0.0)
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

        loss = self.clf_weight * cls_loss + self.distill_weight * (KL_loss + token_distill_loss)  

        return loss, [self.clf_weight * cls_loss, self.distill_weight * KL_loss, self.distill_weight * token_distill_loss]
