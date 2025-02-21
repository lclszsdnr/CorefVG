# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import numpy as np

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .transformer import build_transformer
from util.box_ops import box_cxcywh_to_xyxy,box_iou

## MDETR类
class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,

    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes ,在mdter中不再是物体类别，而是对应的输出token位置（值为256）
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        ## 模型参数和结构的设置
        # query的数目
        self.num_queries = num_queries
        #transform模块，该模块同时包含编码器部分和解码器部分，根据不同的输入决定是编码还是解码
        self.transformer = transformer
        #整个模型中特征的维度
        hidden_dim = transformer.d_model

        # 预测边界框的FFN
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.class_embed = nn.Linear(hidden_dim, 256)
        # self.query_embed = nn.Embedding(100, hidden_dim)
        # self.mask_embed = nn.Embedding(1, hidden_dim)

        # self.isfinal_embed = nn.Linear(hidden_dim, 1)
        # query特征的编码器，生成可学习的编码，即该编码器的权重作为query_embed的值
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 使用卷积核大小为1的2D卷积 将骨干网络中的图片的通道数映射到模型特征维度大小，本质上就是对图像特征的维度变换
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        #获取骨干模块（resnet等）
        self.backbone = backbone
        # 是否对中间编码层的输出计算损失

        # self.isfinal_embed = nn.Linear(hidden_dim, 1)



    # 前向传播
    def forward(self, images_data: NestedTensor, refer_data:NestedTensor, knowl_data:NestedTensor, class_data:NestedTensor=None,refer_smask=None, knowl_smask=None, encode_and_save=True, memory_cache=None):


        ## 如果是进行编码，返回最终编码及相关结果
        if encode_and_save:
            #验证之前并没有编码的结果
            assert memory_cache is None
            # 根据骨干网络对图像进行编码，获取图像特征和图像位置编码，features, pos是列表，其可能包含所有中间层的输出
            vis_features, vis_pos = self.backbone(images_data)

            vis_src, vis_mask = vis_features[-1].decompose()  #[bs, d_backbone, h, w ],[bs, h, w ]
            # query_embed = self.query_embed.weight #[num_query, d_model]
            # mask_embed= self.mask_embed.weight
            #使用多模态编码器获取多模态编码等相关内容
            memory_cache = self.transformer(
                #使用一个二维卷积将骨干网络得到的图像特征维度变为模型维度
                self.input_proj(vis_src), #[bs, d_model, h, w ]
                vis_mask,
                vis_pos[-1],  # [bs,d_model,h,w]
                # query_embed =query_embed,
                # mask_embed = mask_embed,
                refer_data=refer_data,
                knowl_data=knowl_data,

                encode_and_save=True,
            )

            return memory_cache

        ## 进行解码，
        else:
            #验证已经有编码结果了
            assert memory_cache is not None
            #根据编码器输出的信息，进行解码，hs就是query的最终编码
            hs= self.transformer(
                img_memory =memory_cache["img_memory"],
                mask = memory_cache["mask"],
                pos_embed=memory_cache["pos_embed"],
                refer_query_embed=memory_cache["r_query_embed"],
                # refer_mask = memory_cache['refer_mask'],
                encode_and_save=False,
            ) #[num_layers,bs,n_query,d_model]
            # is_final = self.isfinal_embed(hs[ :, :100, :])
            # _, idx = torch.max(is_final.squeeze(), dim=-1)
            #
            # query_list = []
            # for i in range(hs.shape[0]):
            #     query_list.append(hs[i,idx[i],:].unsqueeze(0))
            # hs = torch.cat(query_list,dim=0)
            out = {}
            # outputs_class = self.class_embed(hs)
            #根据解码器输出的hs获取物体边界框
            outputs_coord= self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_boxes_q":outputs_coord,  # [bs,1,4]
                    "q_query": hs,
                    # "outputs_class":outputs_class,
                    # "att": memory_cache['att'],
                    # "retrival_logits": memory_cache['retrival_logits']
                }
            )

            return out #包含各种输出的字典

## 计算对比学习损失
class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=1000):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_query, o_query,labels=None):
        q_query = q_query /q_query.norm(dim = -1 ,keepdim=True)
        o_query = o_query / o_query.norm(dim=-1, keepdim=True)
        logits = (torch.mm(q_query, o_query.transpose(-1,-2)) / 100).log_softmax(-1)
        # loss_ce = -(logits * labels).sum(-1) / 2
        loss_q = F.cross_entropy(logits, labels,dim=1)
        loss_v = F.cross_entropy(logits, labels,dim=2)
        return (loss_q+loss_v)/2




## DETR的损失计算模块，不同的任务有不同需要计算不同的损失
class SetCriterion(nn.Module):


    def __init__(self,  losses, temperature):

        super().__init__()
        ## 设置各参数
        self.losses = losses
        self.temperature = temperature
    def loss_div(self, outputs, targets):
        atts = outputs['att']
        index = targets['index']
        index = index.unsqueeze(-1).repeat(1,1,atts.shape[-1])
        S = atts.detach().cpu().numpy()
        a = torch.gather(atts,1,index)
        a = a / a.norm(dim=-1,keepdim=True)
        similar =  a @ a.transpose(-2,-1)
        a_label = (1 - targets['label'])
        loss_a= torch.sum(((similar * a_label) ** 2).view(-1)) / a_label.count_nonzero()

        b_label = (targets['label'] - torch.eye(targets['label'].shape[-1],device=a_label.device) \
                   .unsqueeze(0).repeat(targets['label'].shape[0],1, 1))
        if b_label.count_nonzero() ==0 :
            loss_b = 0
        else:
            loss_b =  torch.sum((((1-similar)*b_label)   ** 2).view(-1)) /  b_label.count_nonzero()
        losses = {}
        losses['loss_div'] = loss_b +loss_a


        return losses

    def loss_query(self, outputs, targets ):
        target_labels = targets['label']
        query_embed = outputs['query_embed']
        knowl_embed = outputs["knowl_embed"]
        loss = {}

        bs  =query_embed.shape[0]
        contrastive_criterion = ContrastiveCriterion(temperature=self.temperature )


        loss["loss_query"] =contrastive_criterion(query_embed,knowl_embed,target_labels)


        loss["loss_query"] = loss["loss_query"].sum(-1) / bs

        return loss





    ## 计算边界框回归的损失，主要包括l1回归损失和Giou损失
    def loss_boxes(self, outputs, target ):
        target_boxes = target['boxes']
        bs = target_boxes.shape[0]




        src_boxes = outputs["pred_boxes_q"].squeeze(1)

        n = src_boxes .shape[0] / bs
        #计算l1损失
        target_boxes =  target_boxes.repeat_interleave(int(n), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox * outputs['retrival_logits'].reshape(-1,1)
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / bs
        #计算iou损失
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        )

        loss_giou  = loss_giou  * outputs['retrival_logits'].reshape(-1)
        losses["loss_giou"] = loss_giou.sum() / bs
        return losses

    def loss_labels(self, outputs, targets,):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        target_labels = targets['label']

        #将结果先进行softmax计算
        logits = outputs["outputs_class"].log_softmax(-1)  # [bs,num_queries,num_tokens]
        target_labels =  torch.cat([target_labels[:,0].unsqueeze(1),target_labels[:,2:],target_labels[:,0].unsqueeze(1)],dim=1)
        bs =  logits.shape[0]
        loss_ce = -(logits * target_labels).sum(-1) / 2
        loss_ce = loss_ce.sum(-1) / bs

        losses = {"loss_label": loss_ce}

        return losses

    def loss_att(self, outputs, targets, ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        att1 = outputs['att1']
        att2 = outputs['att2']
        bs = att1.shape[0]
        len_refer = att1.shape[1]
        # 将结果先进行softmax计算
        logits= att1 @ att2.transpose(-2,-1)

        # loss_att = torch.diagonal(loss_att,dim1=1,dim2=2)
        labels = torch.arange(len_refer).unsqueeze(0).repeat(bs,1).to(logits.device)
        loss_i = F.cross_entropy(logits, labels, )
        loss_t = F.cross_entropy(logits, labels, )
        loss = (loss_i + loss_t) / 2

        # loss_att = - torch.log( loss_att.sum().sum()/bs/len_refer)

        losses = {"loss_att": loss}

        return losses

    ## 计算各种损失的一个封装函数，loss参数是str，代表要计算哪种损失
    def get_loss(self, loss, outputs, targets,  **kwargs):
        loss_map = {
            "query": self.loss_query,
            "boxes": self.loss_boxes,
            "labels" :self.loss_labels,
            "div" : self.loss_div
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        #调用真正的损失计算函数，计算损失
        return loss_map[loss](outputs, targets,  **kwargs)

    def forward(self, outputs, targets):

        ## Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, ))
        return losses



## 用于预测边界框和token分布的FFN
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    #设置检测物体类别
    device = torch.device(args.device)

    #建立对图像处理的骨干网络，主要包括resnet特征提取网络和位置编码层，输出也是有两个，网络特征和位置编码
    backbone = build_backbone(args)
    #建立transform结构，即包括编码器部分，也包括解码器部分
    transformer = build_transformer(args)
    # 加载一般的MDETR模型
    model = MDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
    )


    weight_dict = {"loss_query": args.query_loss_coef,
                   "loss_bbox": args.bbox_loss_coef,
                   "loss_giou" :args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   # "loss_att" :args.att_loss_coef
                   }



    ##设置损失的种类
    losses = ["boxes"]
    # losses = ["boxes","att"]
    # losses = ["boxes", 'labels']
    # losses = ["query", "boxes", ]
    # losses = ["query" ]

        #建立损失计算模型
    criterion = SetCriterion(
        losses,
        temperature=args.temperature_NCE,
        )
    criterion.to(device)


    return model, criterion,  weight_dict




