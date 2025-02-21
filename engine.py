# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 5.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim
import torch.distributed as distribute
import util.dist as dist
import util.eval_utils as eval_utils
from util.metrics import MetricLogger, SmoothedValue
from util.optim import adjust_learning_rate, update_ema

## 对模型训练一个epoch的函数
def train_one_epoch(
    model: torch.nn.Module, # 模型
    criterion: Optional[torch.nn.Module], #损失计算模块
    weight_dict: Dict[str, float], #各部分损失权重
    data_loader, #数据加载器
    optimizer: torch.optim.Optimizer, #优化器
    device: torch.device, #gpu号
    epoch: int, #当前epoch
    args,  #所有参数
    max_norm: float = 0, #模型梯度的最大范数，在进行梯度裁剪的时候使用
):
    ## 设置模型和各损失计算器都为训练模式
    model.train()
    if criterion is not None:
        criterion.train()


    ## 建立记录器并添加需要记录的参数
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    #模型需要迭代的总次数
    num_training_steps = int(len(data_loader) * args.epochs)
    # 在这里不要被metric_logger.log_every(data_loader, print_freq, header)唬住了，在函数内部返回的还是data_loader
    # 只是对其中的数据进行了分析和打印
    for i,  final_batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        #获取当前迭代次数
        curr_step = epoch * len(data_loader) + i
        ##获取数据
        img_data = final_batch["samples"].to(device)
        # text_data = final_batch['texts'].to(device)
        refer_data = final_batch['refers'].to(device)
        knowl_data = final_batch['captions'].to(device)
        refer_smask = final_batch['refer_special_tokens_mask'].to(device)
        knowl_smask = final_batch['captions_special_tokens_mask'].to(device)

        # batch_size = text_data.tensors.shape[0]
        #
        targets = {}
        targets_boxes = torch.cat([t["boxes"] for t in final_batch['targets']]).to(device)
        # targets_labels = torch.cat([t["label"] for t in final_batch['targets']]).to(device)
        targets['boxes'] = targets_boxes
        # targets['label'] = final_batch['label_bs'].to(device)
        # targets['index'] = final_batch['indexs_bs'].to(device)


        # targets['label'] = targets_labels
        #
        # ## 前向传播获取输出
        # memory_cache = None
        #编码部分，获取编码器输出，主要为文本和图像的编码数据

        memory_cache = model(img_data, refer_data,knowl_data,refer_smask,knowl_smask, encode_and_save=True)
        #解码部分，获取最终解码输出，
        outputs = model(img_data, refer_data,knowl_data,  encode_and_save=False, memory_cache=memory_cache)

        ## 计算各损失值
        loss_dict = {}
        if criterion is not None:

            loss_dict.update(criterion(outputs, targets))

        # distribute.barrier()
        #根据权重系数获取总损失
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        ## reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        ## 优化和反向传播
        optimizer.zero_grad()
        losses.backward()
        #梯度裁剪
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        #根据策略调整学习率
        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        #对模型参数进行指数移动平均值优化

        #将得到的各损失和学习率添加到logger中
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    ## gather the stats from all processes，聚和所有gpu上的值
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## 模型的验证函数
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    device: torch.device,
    args,
):
    ##将各模型设置为测评状态
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for final_batch in metric_logger.log_every(data_loader, 10, header):
        ## 获取数据
        img_data=final_batch["samples"].to(device)
        # text_data = final_batch['texts'].to(device)
        refer_data = final_batch['refers'].to(device)
        knowl_data = final_batch['captions'].to(device)
        refer_smask = final_batch['refer_special_tokens_mask'].to(device)
        knowl_smask = final_batch['captions_special_tokens_mask'].to(device)
        batch_size = refer_data.tensors.shape[0]
        targets = {}
        targets_boxes = torch.cat([t["boxes"] for t in final_batch['targets']]).to(device)
        # targets_labels = torch.cat([t["label"] for t in final_batch['targets']]).to(device)
        targets['boxes'] = targets_boxes
        # targets['label'] = targets_labels
        # targets['label'] = final_batch['label_bs'].to(device)
        # targets['index'] = final_batch['indexs_bs'].to(device)
        ## 前向传播获取输出
        memory_cache = None
        #编码部分，获取编码器输出，主要为文本和图像的编码数据

        memory_cache = model(img_data, refer_data,knowl_data,refer_smask,knowl_smask, encode_and_save=True)
        #解码部分，获取最终解码输出，
        outputs = model(img_data,refer_data,knowl_data,  encode_and_save=False, memory_cache=memory_cache)
        ## 计算损失值
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
        )
        pred_boxes=outputs["pred_boxes_q"].squeeze(1)


        _,index = outputs['retrival_logits'].max(dim=-1)
        pred_boxes = pred_boxes.reshape(batch_size, -1, 4)
        pred_boxes = torch.gather(pred_boxes,dim=1,index=index.reshape(batch_size, 1, 1).repeat(1, 1, 4)).squeeze()

        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, targets["boxes"])
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats




