# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers import RobertaModel, RobertaTokenizerFast
from transformers.models.roberta.configuration_roberta import RobertaConfig


## Transformer类
class Transformer(nn.Module):
    def __init__(
            self,
            d_model=512,  # 模型中特征的维度
            nhead=8,  # 注意力中的头数
            num_encoder_layers=6,  # 编码器部分的transform结构层数
            num_decoder_layers=6,  # 解码器部分的transform结构层数
            dim_feedforward=2048,  # 中间映射维度
            dropout=0.1,  # dropout 概率值
            activation="relu",  # 激活函数种类
            normalize_before=False,  # 决定在激活前进行归一化还是之后
            return_intermediate_dec=False,  # 是否返回每个中间层输出
            pass_pos_and_query=True,  # 决定对初始query和位置编码的利用方式
            text_encoder_type="roberta-base",  # 文本编码器类别
            freeze_text_encoder=False,  # 是否冻结文本编码器
    ):
        super().__init__()
        self.register_parameter('temp', torch.nn.Parameter(torch.ones([]) * 200))
        self.weight_ffn = MLP(768, 512, 1, 2)

        self.weight_ffns = MLP(1, 1, 1, 2)
        self.pass_pos_and_query = pass_pos_and_query
        ## 建立encoder ，负责对输入特征（图像编码特征和文本编码特征）进行自注意力计算
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        ## 建立decoder，先对输入的query进行自注意力计算，然后以query为key，以编码器产生的输入内容编码为value 进行跨注意力计算
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(
            decoder_layer, 6, decoder_norm, return_intermediate=return_intermediate_dec
        )


        res_layer = TransformerResLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        resloter_norm = nn.LayerNorm(d_model)
        self.resloter = TransformerDecoder(
            res_layer, 3, resloter_norm, return_intermediate=return_intermediate_dec
        )

        # 对模型参数进行初始化
        self._reset_parameters()

        # 设置文本编码器，默认为roberta
        self.text_encoder = RobertaModel.from_pretrained('./models/cache_dir/ssss')
        self.retriver = RobertaModel.from_pretrained('./models/cache_dir/ssss')

        self.text_id_embeddings = self.text_encoder.embeddings.word_embeddings
        ## 根据  freeze_text_encoder 决定是否冻结文本编码器
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        ## 将文本编码器输出的特征维度转换到模型要求的维度
        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

        self.len_triver = 6
        self.topk = 5

    ## 对模型中维度大于1的参数 进行初始化
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    ## transformer中包含解码器和编码器两部分，有的时候是进行编码，有的时候是进行解码，
    # 所以在forward输入输出中有些混乱，有些输入就是自己的编码器产生的
    def forward(
            self,
            src=None,  # [bs,d_model,H,W]
            mask=None,  # [bs,H,W]
            pos_embed=None,
            refer_query_embed=None,
            refer_mask=None,
            knowl_mask=None,
            refer_data=None,
            knowl_data=None,  #
            query_embed=None,

            encode_and_save=True,  # 是否是编码器还是解码器
            img_memory=None,
            knowl_memory=None,  #
    ):
        ##进行编码
        if encode_and_save:
            ## flatten NxCxHxW to HWxNxC ，C=d_model

            bs, c, h, w = src.shape
            # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            src = src.flatten(2).permute(2, 0, 1)  # [HW,bs,d_model]
            mask = mask.flatten(1)  # [bs,hw]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            len_src = src.shape[0]
            if self.pass_pos_and_query:
                pass
            else:
                src, pos_embed = src + 0.1 * pos_embed, None
            len_refer = refer_data.mask.shape[1]
            len_know = knowl_data.mask.shape[1]

            # class_mask = class_data.mask.ne(1).bool()


            r_masks = torch.cat([refer_data.mask, knowl_data.mask], dim=1
                                ).unsqueeze(1).repeat(1, len_refer, 1)

            k_masks = torch.cat([refer_data.mask, knowl_data.mask], dim=1
                                ).unsqueeze(1).repeat(1, len_know, 1)

            text_tensors = torch.cat([refer_data.tensors, knowl_data.tensors], dim=1)
            text_masks = torch.cat([r_masks, k_masks], dim=1)
            text_output = self.retriver(input_ids=text_tensors,
                                            attention_mask=text_masks,
                                            output_attentions=True,
                                            )
            encoded_text = text_output.last_hidden_state

            t = encoded_text/ encoded_text.norm(dim=2, keepdim=True)
            rk_att = t[:, :len_refer] @ t[:, len_refer:].transpose(-1, -2)  # [bs,len_r,len_know]
            refer_weights = self.weight_ffn(encoded_text[:, :len_refer])  # [bs,len_r,1]

            rk_att = refer_weights * rk_att
            rk_atts = rk_att.sum(1, keepdims=True)
            rk_atts =  self.weight_ffns(rk_atts.transpose(1,2)).transpose(1,2)
            kernel = torch.ones(self.len_triver).to(rk_att.device)
            # 使用卷积操作计算最后一维相邻6个元素之和
            output = F.conv1d(rk_atts, kernel.view(1, 1, -1),stride=max(1,int(self.len_triver/4)) )
            _,index = torch.topk(output, k=self.topk)
            logits = ((_ / _.sum(dim=-1,keepdim=True))*self.temp).softmax(dim=-1)


            relvant_knowl_id = knowl_data.tensors.unfold(dimension=-1,size=self.len_triver,step=max(1,int(self.len_triver/4)))
            relvant_knowl_id = torch.gather(relvant_knowl_id,1,index.transpose(1,2)
                                            .repeat(1,1,self.len_triver)).reshape(-1,self.len_triver)

            relvant_knowl_mask = knowl_data.mask.unfold(dimension=-1,size=self.len_triver,step=int(self.len_triver/2))
            relvant_knowl_mask = torch.gather(relvant_knowl_mask,1,index.transpose(1,2)
                                              .repeat(1,1,self.len_triver)).reshape(-1,self.len_triver)

            refer_data_ids = refer_data.tensors.unsqueeze(1).repeat(1,self.topk,1).reshape(-1,len_refer)
            refer_mask = refer_data.mask.unsqueeze(1).repeat(1,self.topk,1).reshape(-1,len_refer)
            knowl_data_ids = knowl_data.tensors.unsqueeze(1).repeat(1,self.topk,1).reshape(-1,len_know)
            knowl_mask = knowl_data.mask.unsqueeze(1).repeat(1,self.topk,1).reshape(-1,len_know)


            #
            # refer_data_ids = torch.cat([refer_data_ids[:, 0:1], relvant_knowl_id,
            #                             torch.ones_like(refer_data_ids[:, 0:1]) * 6,
            #
            #                             ], dim=1)
            # refer_mask = torch.ones_like(knowl_mask[:, :self.len_triver + 2])
            # r_masks = torch.cat([refer_mask,
            #                      knowl_mask], dim=1
            #                     ).unsqueeze(1).repeat(1, self.len_triver+2, 1)


            refer_data_ids = torch.cat([refer_data_ids[:,0:1],relvant_knowl_id,
                                        torch.ones_like(refer_data_ids[:,0:1])*6,
                                        refer_data_ids[:, 1:]
                                        ],dim=1)
            refer_mask = torch.cat([refer_mask[:,0:-1],torch.ones_like(refer_mask[:,0:1]),
                                 relvant_knowl_mask,refer_mask[:,-1:]], dim=1)
            r_masks = torch.cat([refer_mask,
                                 knowl_mask], dim=1
                                ).unsqueeze(1).repeat(1, self.len_triver+len_refer+1, 1)




            # refer_data_ids = relvant_knowl_id
            # refer_mask = torch.zeros_like(relvant_knowl_mask)
            # r_masks = torch.cat([refer_mask,
            #                      knowl_mask], dim=1
            #                     ).unsqueeze(1).repeat(1,self.len_triver, 1)

            k_masks = torch.cat([refer_mask,
                                 knowl_mask], dim=1
                                ).unsqueeze(1).repeat(1, len_know, 1)
            text_tensors = torch.cat([
                                     refer_data_ids,
                                      knowl_data_ids], dim=1)

            text_masks = torch.cat([r_masks, k_masks], dim=1)
            text_output = self.text_encoder(input_ids=text_tensors,
                                            attention_mask=text_masks,
                                            output_attentions=True,
                                            )
            encoded_text = text_output.last_hidden_state


            text_memory = self.resizer(encoded_text).transpose(0, 1)
            refer_memory = text_memory[:-len_know]
            knowl_memory = text_memory[-len_know:]
            knowl_mask = knowl_mask.ne(1).bool()
            refer_mask = refer_mask.ne(1).bool()


            src = src.repeat_interleave(self.topk,dim=1)
            pos_embed = pos_embed.repeat_interleave(self.topk,dim=1)
            mask = mask.repeat_interleave(self.topk,dim=0)

            src_r = torch.cat([src, knowl_memory ],
                              dim=0)  # [HW+batch_max_sequence_len,bs,d_model]
            mask_r = torch.cat([mask, knowl_mask],
                               dim=1)  # [bs,HW+batch_max_sequence_len]
            pos_embed_r = torch.cat([pos_embed,
                                     # torch.zeros_like(relevent_knowl),
                                     torch.zeros_like(knowl_memory), ], dim=0)
            src_r, kv_att = self.encoder.layers[0](src_r, src_key_padding_mask=mask_r,
                                                   pos=pos_embed_r)
            refer_memory = self.resloter.layers[0](refer_memory,  src_r[len_src:], src_r[len_src:],

                                                   tgt_key_padding_mask=refer_mask,
                                                   memory_key_padding_mask=knowl_mask,
                                                   )


            src_r, kv_att = self.encoder.layers[1](src_r, src_key_padding_mask=mask_r,
                                                   pos=pos_embed_r)

            refer_memory = self.resloter.layers[1](refer_memory,  src_r[len_src:], src_r[len_src:],

                                                   tgt_key_padding_mask=refer_mask,
                                                   memory_key_padding_mask=knowl_mask,
                                                   )

            #
            src_r, kv_att = self.encoder.layers[2](src_r, src_key_padding_mask=mask_r,
                                                   pos=pos_embed_r)
            refer_memory = self.resloter.layers[2](refer_memory, src_r[len_src:],src_r[len_src:],

                                                   tgt_key_padding_mask=refer_mask,
                                                   memory_key_padding_mask=knowl_mask,
                                                   )


            # relevent_knowl = src_r[:6]
            # refer_memory = torch.cat([refer_memory,relevent_knowl],dim=0)
            # refer_mask = torch.cat([refer_mask,knowl_mask[:,:6]],dim=1)

            # src_n = torch.cat([src,refer_memory,knowl_memory],
            #                  dim=0)  # [HW+batch_max_sdequence_len,bs,d_model]
            # mask_n = torch.cat([mask,refer_mask,knowl_mask],
            #                   dim=1)  # [bs,HW+batch_max_sequence_len]
            # pos_embed_n = torch.cat([pos_embed,torch.zeros_like(refer_memory),
            #                         ], dim=0)
            #
            #
            # src_n, kv_att = self.encoder.layers[0](src_n, src_key_padding_mask=mask_n,
            #                                        pos=pos_embed_n)
            # src_n, kv_att = self.encoder.layers[1](src_n, src_key_padding_mask=mask_n,
            #                                        pos=pos_embed_n)
            # src_n, kv_att = self.encoder.layers[2](src_n, src_key_padding_mask=mask_n,
            #                                        pos=pos_embed_n)

            #
            # refer_memory = ((1-refer_weights)*src_n[len_src:len_refer + len_src] +
            #                 refer_weights*src_r[len_src:len_refer + len_src])
            # src = ((1 - refer_weights).sum() / (refer_weights.sum() + (1 - refer_weights).sum())) * src_n[:len_src] + \
            #       (refer_weights.sum() / (refer_weights.sum() + (1 - refer_weights).sum())) * src_r[:len_src]
            # pos_embed = pos_embed[:len_src]
            # mask = mask[:, :len_src]

            src = src_r[:len_src]
            src = torch.cat([src, refer_memory],
                            dim=0)  # [HW+batch_max_sequence_len,bs,d_model]
            mask = torch.cat([mask, refer_mask],
                             dim=1)  # [bs,HW+batch_max_sequence_len]
            pos_embed = torch.cat([pos_embed, torch.zeros_like(refer_memory),
                                   ], dim=0)

            src, eatt = self.encoder.layers[3](src, src_key_padding_mask=mask,
                                               pos=pos_embed)
            src, eatt = self.encoder.layers[4](src, src_key_padding_mask=mask,
                                               pos=pos_embed)
            src, eatt = self.encoder.layers[5](src, src_key_padding_mask=mask,
                                               pos=pos_embed)



            img_memory = src[:len_src]
            pos_embed = pos_embed[:len_src]
            mask = mask[:, :len_src]
            refer_query_embed = refer_memory
            # src[:len_src]

            if self.pass_pos_and_query:
                r_tgt = torch.zeros_like(refer_query_embed)
            else:
                r_tgt, refer_query_embed, = refer_query_embed, None

            memory_cache = {
                "img_memory": img_memory,  # [HW+batch_max_sequence_len,bs,d_model]，多模态编码器输出的 综合特征向量
                "mask": mask,  # [bs,HW+batch_max_sequence_len] ，表明哪些元素是真实的，哪些是pad的
                "pos_embed": pos_embed,
                ##[HW+batch_max_sequence_len,bs,d_model]，主要是对图像中像素点的位置编码，batch_max_sequence_len对应的文本编码部分都是0
                "r_query_embed": refer_query_embed,  # [n_query,bs,d_model],
                "attention_shape": (h, w),
                "knowl_memory": knowl_memory,
                "rk_att" :rk_att,
                "refer_mask": refer_mask,
                "knowl_mask": knowl_mask,
                "retrival_logits":logits.squeeze()

            }

            return memory_cache
        ## 进行解码
        else:
            ## 这里的设置就是选择是否在编码、解码中还反复利用pos_embed,和query_embed
            # 是的话就令tgt为0，之后再根据query计算，src也在之后也pos_embded多次相加
            # 不是的话，就只是将pos_embed在最开始以0.1的比重加到src中，也是将query直接赋值给tgt，之后pos_embed和query直接置零，不再使用
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(refer_query_embed)
            else:
                src, tgt, refer_query_embed, pos_embed = src + 0.1 * pos_embed, refer_query_embed, None, None
            assert img_memory.shape[1] == tgt.shape[1]
            hs, datt = self.decoder(
                tgt,
                img_memory,
                # tgt_key_padding_mask = refer_mask,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                query_pos=refer_query_embed,
            )
            hs = hs.transpose(1, 2)

            return hs[-1, :, 0, :]  # [6,bs,n_query,d_model]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = layer(x)

            else:
                x = F.sigmoid(layer(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 将encoder_layer层建立 num_layers次
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):

        output = src
        atts = []
        # 按顺序在所有层中进行计算
        for layer in self.layers:
            output, att = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            atts.append(att)
        # 归一化
        if self.norm is not None:
            output = self.norm(output)

        return output, atts


## Decoder模块
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 将decoder_layer层建立 num_layers次
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            tgt,
            memory,
            text_memory=None,  # [batch_max_sequence_len,bs,d_model],多模态文本特征
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            text_memory_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        atts = []
        intermediate = []
        ##按顺序在所有层中进行计算
        for layer in self.layers:
            output, att = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            atts.append(att)
            # 如果返回中间层，则将每个层的输出加载到列表中
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # 将最后一层的输出归一化
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        # 如果返回中间层，则将每个层的输出堆叠到一起
        if self.return_intermediate:
            return torch.stack(intermediate), torch.cat(atts)

        return output


## Transformer的Encoder模块层，负责对输入特征（图像编码特征和文本编码特征）进行自注意力计算
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        ## 各注意力层、线性层及归一化层和激活函数层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # 将内容向量和位置向量相加，类似于bert中不同编码方式的相加
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    ##常见的transform层前向传播，只是根据normalize_before建立了两个前向传播
    # post是先计算再归一化，pre是先归一化再计算
    def forward_post(
            self,
            src,  # [HW+batch_max_sequence_lenbs,d_model]，图像特征和文本特征的拼接
            src_mask: Optional[Tensor] = None,  # 无
            src_key_padding_mask: Optional[Tensor] = None,  # [HW+batch_max_sequence_len,bs] 对应的元素真实情况，即是真实的还是pad的
            pos: Optional[Tensor] = None,  # [HW+batch_max_sequence_len,bs,d_model] 位置编码，文本部分是0，主要是记录图像位置编码
    ):
        q = k = self.with_pos_embed(src, pos)  # [HW+batch_max_sequence_len,bs,d_model]
        # attn_mask 与q,k都有关，指明两者之间的对应关系，key_padding_mask只于k有关，是k本身的问题
        # 前者维度为 （bs, key batch_max_sequence_len,value squence_lengtg）,后者维度为（bs,value squence_lengtg）
        src2, att = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, att  # [HW+batch_max_sequence_len,bs,d_model]

    ## 与上面的类似
    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    ## 真正的前向传播，选择前面两个中的一个即可
    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


## Transformer的Dencoder模块层，先对输入的query进行自注意力计算，然后以query为key，以编码器产生的输入内容编码为value 进行跨注意力计算
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        ## 各注意力层、线性层及归一化层和激活函数层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    ## 带有自注意力计算和跨模态注意力计算的解码器前向传播过程
    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
            self,
            tgt,  # [n_query,bs,d_model] ,是query在不断解码后的中间值，其初始值可以是query也可以是0
            memory,  # [HW+batch_max_sequence_len,bs,d_model],多模态特征
            text_memory=None,  # [batch_max_sequence_len,bs,d_model],多模态文本特征
            tgt_mask: Optional[Tensor] = None,  # 未知
            memory_mask: Optional[Tensor] = None,  # 未知
            text_memory_key_padding_mask: Optional[Tensor] = None,  # [bs,batch_max_sequence_len] 文本编码的mask
            tgt_key_padding_mask: Optional[Tensor] = None,  # 未知
            memory_key_padding_mask: Optional[Tensor] = None,
            # #[bs,HW+batch_max_sequence_len] ，表明memory哪些元素是真实的，哪些是pad的
            pos: Optional[Tensor] = None,  # [HW+batch_max_sequence_len,bs,d_model],位置编码，当memory已经在初始时就加载了位置编码时为0
            query_pos: Optional[Tensor] = None,  # query的编码，在 tgt初始值为query时值为0
    ):
        # q，k在最初始时值就是query，无论是在哪种设置下，也对应了解码器结构
        q = k = self.with_pos_embed(tgt, query_pos)

        ## Self attention,tgt(也可以视为query)进行自注意力计算
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # tgt =torch.cat([(tgt + self.dropout1(tgt2))[0].unsqueeze(0),tgt[1:]],dim=0)
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        ## Cross attention to image，query与多模态输入值的交互
        tgt2, att = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN，最后的一个前向传播网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, att  # [n_query,bs,d_model]

    ## 与上面的类似
    def forward_pre(
            self,
            tgt,
            memory,
            text_memory=None,  # [batch_max_sequence_len,bs,d_model],多模态文本特征
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            text_memory_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    ## 真正的前向传播，选择前面两个中的一个即可
    def forward(
            self,
            tgt,
            memory,
            text_memory=None,  # [batch_max_sequence_len,bs,d_model],多模态文本特征
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            text_memory_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


## Transformer的Dencoder模块层，先对输入的query进行自注意力计算，然后以query为key，以编码器产生的输入内容编码为value 进行跨注意力计算
class TransformerResLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        ## 各注意力层、线性层及归一化层和激活函数层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    ## 真正的前向传播，选择前面两个中的一个即可
    def forward(
            self,
            tgt,
            k_memory,
            v_memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ):
        ## Self attention,tgt(也可以视为query)进行自注意力计算
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # tgt =torch.cat([(tgt + self.dropout1(tgt2))[0].unsqueeze(0),tgt[1:]],dim=0)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        ## Cross attention to image，query与多模态输入值的交互
        tgt2 = self.cross_attn_image(
            query=tgt,
            key=k_memory,
            value=v_memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        # A = self.cross_attn_image.in_proj_weight.grad()
        tgt = self.norm2(tgt)

        return tgt  # [n_query,bs,d_model]


## 通过一个线性层转换，改变特征维度
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


## 将module复制n个，并放入ModuleList中
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


## 使用 Transformer类建立 transform
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
    )


## 根据名字选取对应的激活函数
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


## 通过一个线性层转换，改变特征维度
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


## 将module复制n个，并放入ModuleList中
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


## 使用 Transformer类建立 transform
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
    )


## 根据名字选取对应的激活函数
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
