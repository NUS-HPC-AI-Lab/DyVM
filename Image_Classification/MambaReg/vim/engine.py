# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DyVMLoss_Token_Block, DyVMLoss_Block_Only, DyVMLoss_Token_Only
import utils


#########################
# token block layer
#########################

# def train_one_epoch(model: torch.nn.Module, criterion: DyVMLoss_Token_Block_Layer,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('token_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('block_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('layer_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('token_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('block_keep_ratio', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('layer_keep_ratio', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
    
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
        
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
            
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
            
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
         
#         with amp_autocast():
#             outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)

#             if not args.cosub:
#                 loss, loss_parts = criterion(samples, outputs, targets)
#             else:
#                 raise NotImplementedError

#         if args.if_nan2num:
#             with amp_autocast():
#                 loss = torch.nan_to_num(loss)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             if args.if_continue_inf:
#                 optimizer.zero_grad()
#                 continue
#             else:
#                 sys.exit(1)

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         if isinstance(loss_scaler, timm.utils.NativeScaler):
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         else:
#             loss.backward()
#             if max_norm != None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(cls_loss=loss_parts[0].item())
#         metric_logger.update(token_pruning_loss=loss_parts[1].item())
#         metric_logger.update(block_pruning_loss=loss_parts[2].item())
#         metric_logger.update(layer_pruning_loss=loss_parts[3].item())
#         metric_logger.update(cls_kl_loss=loss_parts[4].item())
#         metric_logger.update(token_kl_loss=loss_parts[5].item())
#         metric_logger.update(block_keep_ratio=outputs[-2].mean().item())
#         metric_logger.update(layer_keep_ratio=outputs[-1].mean().item())
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(data_loader, model, device, amp_autocast, model_name, output_dir_sel_policy=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for images, target in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with amp_autocast():
#             output, block_policy, layer_policy = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#         metric_logger.meters['block_keep_ratio'].update(block_policy.mean().item(), n=batch_size)
#         metric_logger.meters['layer_keep_ratio'].update(layer_policy.mean().item(), n=batch_size)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} block_keep_ratio {block_keep_ratio.global_avg:.3f} layer_keep_ratio {layer_keep_ratio.global_avg:.3f}'
#         .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, block_keep_ratio=metric_logger.block_keep_ratio, layer_keep_ratio=metric_logger.layer_keep_ratio))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#########################
# token block
#########################

def train_one_epoch_token_block(model: torch.nn.Module, criterion: DyVMLoss_Token_Block,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('token_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('block_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('token_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('block_keep_ratio', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples)

            if not args.cosub:
                loss, loss_parts = criterion(samples, outputs, targets)
            else:
                raise NotImplementedError

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(cls_loss=loss_parts[0].item())
        metric_logger.update(token_pruning_loss=loss_parts[1].item())
        metric_logger.update(block_pruning_loss=loss_parts[2].item())
        metric_logger.update(cls_kl_loss=loss_parts[3].item())
        metric_logger.update(token_kl_loss=loss_parts[4].item())
        metric_logger.update(block_keep_ratio=outputs[-1].mean().item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_token_block(data_loader, model, device, amp_autocast, model_name, output_dir_sel_policy=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output, block_policy = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['block_keep_ratio'].update(block_policy.mean().item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} block_keep_ratio {block_keep_ratio.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, block_keep_ratio=metric_logger.block_keep_ratio))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#########################
# token layer
#########################
# def train_one_epoch(model: torch.nn.Module, criterion: DyVMLoss_Token_Layer,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('token_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('layer_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('token_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('layer_keep_ratio', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
    
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
        
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
            
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
            
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
         
#         with amp_autocast():
#             outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)

#             if not args.cosub:
#                 loss, loss_parts = criterion(samples, outputs, targets)
#             else:
#                 raise NotImplementedError

#         if args.if_nan2num:
#             with amp_autocast():
#                 loss = torch.nan_to_num(loss)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             if args.if_continue_inf:
#                 optimizer.zero_grad()
#                 continue
#             else:
#                 sys.exit(1)

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         if isinstance(loss_scaler, timm.utils.NativeScaler):
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         else:
#             loss.backward()
#             if max_norm != None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(cls_loss=loss_parts[0].item())
#         metric_logger.update(token_pruning_loss=loss_parts[1].item())
#         metric_logger.update(layer_pruning_loss=loss_parts[2].item())
#         metric_logger.update(cls_kl_loss=loss_parts[3].item())
#         metric_logger.update(token_kl_loss=loss_parts[4].item())
#         metric_logger.update(layer_keep_ratio=outputs[-1].mean().item())
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(data_loader, model, device, amp_autocast, model_name, output_dir_sel_policy=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for images, target in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with amp_autocast():
#             output, block_policy, layer_policy = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#         metric_logger.meters['layer_keep_ratio'].update(layer_policy.mean().item(), n=batch_size)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} layer_keep_ratio {layer_keep_ratio.global_avg:.3f}'
#         .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, layer_keep_ratio=metric_logger.layer_keep_ratio))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



#########################
# block only
#########################

def train_one_epoch_block_only(model: torch.nn.Module, criterion: DyVMLoss_Block_Only,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('block_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, 
                           if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss, loss_parts = criterion(samples, outputs, targets)
            else:
                raise NotImplementedError

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        dir_sel_policy = outputs[1]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(cls_loss=loss_parts[0].item())
        metric_logger.update(block_pruning_loss=loss_parts[1].item())
        metric_logger.update(cls_kl_loss=loss_parts[2].item())
        metric_logger.update(block_keep_ratio=dir_sel_policy.mean().item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_block_only(data_loader, model, device, amp_autocast, model_name):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output, dir_sel_policy = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['block_keep_ratio'].update(dir_sel_policy.mean().item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} block_keep_ratio {block_keep_ratio.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, block_keep_ratio=metric_logger.block_keep_ratio))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#########################
# token only
#########################

def train_one_epoch_token_only(model: torch.nn.Module, criterion: DyVMLoss_Token_Only,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('token_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('token_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, 
                           if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss, loss_parts = criterion(samples, outputs, targets)
            else:
                raise NotImplementedError

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(cls_loss=loss_parts[0].item())
        metric_logger.update(token_pruning_loss=loss_parts[1].item())
        metric_logger.update(cls_kl_loss=loss_parts[2].item())
        metric_logger.update(token_kl_loss=loss_parts[3].item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_token_only(data_loader, model, device, amp_autocast, model_name):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#########################
# layer only
#########################

# def train_one_epoch_layer_only(model: torch.nn.Module, criterion: DyVMLoss_Layer_Only,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('layer_pruning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('cls_kl_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
    
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
        
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
            
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
            
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
         
#         with amp_autocast():
#             outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, 
#                            if_random_token_rank=args.if_random_token_rank)
#             # outputs = model(samples)
#             if not args.cosub:
#                 loss, loss_parts = criterion(samples, outputs, targets)
#             else:
#                 raise NotImplementedError

#         if args.if_nan2num:
#             with amp_autocast():
#                 loss = torch.nan_to_num(loss)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             if args.if_continue_inf:
#                 optimizer.zero_grad()
#                 continue
#             else:
#                 sys.exit(1)

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         if isinstance(loss_scaler, timm.utils.NativeScaler):
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)
#         else:
#             loss.backward()
#             if max_norm != None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         layer_policy = outputs[1]
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(cls_loss=loss_parts[0].item())
#         metric_logger.update(layer_pruning_loss=loss_parts[1].item())
#         metric_logger.update(cls_kl_loss=loss_parts[2].item())
#         metric_logger.update(layer_keep_ratio=layer_policy.mean().item())
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate_layer_only(data_loader, model, device, amp_autocast, model_name):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for images, target in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with amp_autocast():
#             output, layer_policy = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#         metric_logger.meters['layer_keep_ratio'].update(layer_policy.mean().item())
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} layer_keep_ratio {layer_keep_ratio.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, layer_keep_ratio=metric_logger.layer_keep_ratio))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}