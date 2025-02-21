# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from timm.models import create_model
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from .position_encoding import build_position_encoding

## 冻结的批归一化模块，其实根本没有实现归一化，是为了对应torchversion中的的结构，
 # 在不想使用归一化时就使用该模块替换
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 在模型类成员变量中设置的tensor张量，无法保存到state_dict中，也无法根据模型的to函数转移到gpu中
        # 需要使用register进行登记，才能解决以上两个问题
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # 重写 _load_from_state_dict 函数
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    ## 可以看到没有进行归一化
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

## 骨干层基类，只需要将需要的骨干模型输入即可
class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        ## 当train_backbone为 fasle 或者 backbone中没有layer2\3\4是，将backbone 中参数设置为不需要梯度
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        ## 设置返回层名词字典,是IntermediateLayerGetter的的第二个参数
        '''
            a dict containing the names of the modules for which the 
            activations will be returned asthe key of the dict, and 
            the value of the dict is the name of the returned activation
        '''
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": 0}
        #IntermediateLayerGetter是从模型中返回中间层的一个模块包装器，backbone就是模型，
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # num_channels 就相当于特征维度，图像中每个点的维度从3扩展到了num_channels，每个点就相当于一个文字
        self.num_channels = num_channels

    def forward(self, tensor_list):
        ##运行模型获取输出，xs是一个迭代器，其中元素为键值对，
         # key为return_layers中设置的层简短名（0，1，2，3），value为对应层的输出值
        xs = self.body(tensor_list.tensors)
        ## 将获得的输出封装到字典中
        out = OrderedDict()
        for name, x in xs.items():
            #interpolate是数组上\下采样操作，在科学合理的改变数组的尺寸大小同时尽量保证数据完整
            #第一个参数input是需要进行采样处理的数组，size是输出空间的大小，这里应该是下采样，因为
            #图像大小随着卷积的过程在减小，对应的mask也需要对应的减小。
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out #dict(name,NestedTensor(x, mask))

## 真正的设置了模型的骨干层
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        #直接按照名字从torchversion的模型库中加载骨干模型，replace_stride_with_dilation决定是否使用膨胀卷积，
        # 注意是加载的预训练的，所以正则化层直接使用冻结的就可以
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], pretrained=True, norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

## 设置了group数目为32的组归一化模块
class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)

## 使用组归一化的resnet骨干网络
class GroupNormBackbone(BackboneBase):
    """ResNet backbone with GroupNorm with 32 channels."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        #还是从torchversion.models中加载骨干网络，只是使用群归一化
        name_map = {
            "resnet50-gn": ("resnet50", "/checkpoint/szagoruyko/imagenet/22014122/checkpoint.pth"),
            "resnet101-gn": ("resnet101", "/checkpoint/szagoruyko/imagenet/22080524/checkpoint.pth"),
        }
        backbone = getattr(torchvision.models, name_map[name][0])(
            replace_stride_with_dilation=[False, False, dilation], pretrained=False, norm_layer=GroupNorm32
        )
        ## 加载缓存权重到模型中
        checkpoint = torch.load(name_map[name][1], map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)
        # 根据模型大小设置channels数目
        num_channels = 512 if name_map[name][0] in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

## 替换函数，将模型中的普通bn都转换为冻结的bn
def replace_bn(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight)
            frozen.bias.data.copy_(bn.bias)
            frozen.running_mean.data.copy_(bn.running_mean)
            frozen.running_var.data.copy_(bn.running_var)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_bn(ch, n)

## 设置了group数目为8的组归一化模块
class GN_8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gn = torch.nn.GroupNorm(8, num_channels)

    def forward(self, x):
        return self.gn(x)

class TimmBackbone(nn.Module):
    def __init__(self, name, return_interm_layers, main_layer=-1, group_norm=False):
        super().__init__()
        backbone = create_model(name, pretrained=False, in_chans=3, features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            replace_bn(backbone)
        num_channels = backbone.feature_info.channels()[-1]
        self.body = backbone
        self.num_channels =  num_channels
        self.interm = return_interm_layers
        self.main_layer = main_layer

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        if not self.interm:
            xs = [xs[self.main_layer]]
        out = OrderedDict()
        for i, x in enumerate(xs):
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[f"layer{i}"] = NestedTensor(x, mask)
        return out

## 就是一个序列层，将图像特征提取骨干模型和编码层放在一起
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        #根据骨干模型获取图像特征编码
        xs = self[0](tensor_list)
        out = []
        pos = []
        '''
        xs是骨干网络中一些中间层的输出组成的字典，key就是中间层的层名，value是NestedTensor，
        NestedTensor中包含两部分，一部分是原有中间层的输出，一部分是该输出对应的mask值,维度都为bs, c, h, w 
        '''
        for name, x in xs.items():
            out.append(x)
            # position encoding
            '''
            根据图像特征获取位置编码，图像位置编码的主要作用是对图像的真实区域进行编码，因为虽然每张图片的特征大小都一致，
            但是本质上有的图片大，有的图片小，图片小的虽然特征和大的如图特征一样，但是特征中只有一部分是真实图像数据，其
            他地方是补零的无用数据，特征编码相当于就是记录着特征中哪些区域是真实图像区域。
            '''
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

#建立 骨干结构
def build_backbone(args):
    #获取位置编码模块
    position_embedding = build_position_encoding(args)
    #根据参数中的 lr_backbone 决定是否训练backbone
    train_backbone = args.lr_backbone > 0
    #根据参数中的 masks 决定是否返回中间层值
    return_interm_layers = True
    # 根据参数中的backbone 决定使用的backbone类别
    if args.backbone[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            args.backbone[len("timm_") :],
            return_interm_layers,
            main_layer=-1,
            group_norm=True,
        )
    elif args.backbone in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    ## 将特征提取骨干网络和位置编码层结合在一起形成 对图像生成编码的model
    model = Joiner(backbone, position_embedding)
    #num_channels 就相当于特征维度，图像中每个点的维度从3扩展到了num_channels，每个点就相当于一个文字
    model.num_channels = backbone.num_channels
    return model
