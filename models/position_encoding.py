# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn

'''
通用的三角函数位置编码
这种编码方式可以令每个位置都有确定的编码，而且不受句子长度的影响，编码范围也是有界的（不会因为句子加长而范围增大）
且相邻位置的编码相似度高，远离位置的编码相似度低。
对于位置pos的编码，其总维度为n,对于每个维度上的值，若该维度位置为偶数2i，该维度上的值为sin(pos/10000** (2i/n))
若为奇数2i+1，则值为cos(pos/10000** (2i/n)).
所以最终pos的编码为[sin(pos/10000** (0/n)),cos(pos/10000** (0/n)),sin(pos/10000** (2/n)),cos(pos/10000** (2/n)),...]
虽然sin和cos是周期函数，但是不同个的sin和cos值组合在一起，则令不同的位置编码一定不同。

'''
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        #位置编码维度
        self.num_pos_feats = num_pos_feats
        #t超参数
        self.temperature = temperature
        ## 是否正则化的标记和 正则化比例 ，这里的正则化就是将值变为（0~sclae）范围
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        # tensor_list应该是DTER的Nestedtensor结构
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        ##通过累加获取各元素的真实行、列位置标签
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        ## 正则化，每个值都除以对应最大值后 *scale比例，正则化到（0~sclae）范围
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        ## 对应公式中的10000** (2/n))
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        #0::2 代表从第0位置开始，每隔2个取一个值 ，flatten为扁平化，可以把m~n维的数据拉到一个维度
        #这种先stack再flatten可以仍然保持 原有的顺序
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  #[b,d,h,w]

#可学习的位置编码，就是直接使用nn.embedding 进行编码，也是常用的方式，但是这种有长度限制
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)#[w]
        j = torch.arange(h, device=x.device)#[h]
        x_emb = self.col_embed(i)#[w,256]
        y_emb = self.row_embed(j)#[h,256]
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),#[h,w,256]
                    y_emb.unsqueeze(1).repeat(1, w, 1),#[h,w,256]
                ],
                dim=-1,
            )#[h,w,512]
            .permute(2, 0, 1)#[512,h,w]
            .unsqueeze(0)#[1,512,h,w]
            .repeat(x.shape[0], 1, 1, 1) #[bs,512,h,w]
        )
        return pos #[bs,512,h,w]

## 建立并返回位置编码模块，注意对图片的位置编码是对图像特征每个像素点的位置进行编码，所以其大小为[bs,d,h,w],d为编码维度
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
