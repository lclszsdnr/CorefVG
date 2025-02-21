# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
import torch.nn as  nn
import util.dist as dist
import util.misc as utils
from dataset import build_dataset
from torch.utils.tensorboard import SummaryWriter
from engine import evaluate, train_one_epoch
from models import build_model

from transformers import logging
logging.set_verbosity_warning()

## 该函数的作用是获设置模型运行时需要用到的参数，并设置一些默认值
def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--data_root", default='./dataset/sk_vg')
    parser.add_argument('--max_query_len', default=15, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--max_knowledge_len', default=128, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--imsize', default=640, type=int, help='image size')

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")


    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=5e-6, type=float)
    parser.add_argument("--text_encoder_lr", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr_drop", default=20, type=int)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="multistep",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )

    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        # default="timm_tf_efficientnet_b5_ns",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer'tokenzier attentions",
    )
    parser.add_argument("--num_queries", default=200, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Loss

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )


    # Loss coefficients
    parser.add_argument("--query_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--label_loss_coef", default=1, type=float)
    parser.add_argument("--div_loss_coef", default=5, type=float)



    # Run specific
    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default='', help="resume from checkpoint")
    # parser.add_argument("--load", default="/media/team/data/CODE/slef_code/mdetr/outputs/retriver2/BEST_checkpoint.pth", help="resume from checkpoint")
    parser.add_argument("--load", default="./models/cache_dir/pretrained_resnet101_checkpoint.pth", help="resume from checkpoint")

    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", default=True,action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser


def main(args):
    writer = SummaryWriter(args.output_dir)
    # Init distributed mode，设置模型并行
    dist.init_distributed_mode(args)

    #设置当前gpu
    device = torch.device(args.device)
    #获取输出路径
    output_dir = Path(args.output_dir)

    ## fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    ## Build the model， criterion为损失计算模块， weight_dict中包含各损失的权重系数
    model, criterion,  weight_dict = build_model(args)
    model.to(device)
    # torch.save(model.state_dict(),'ini.pth')
    ## 进行分布式设置
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True,broadcast_buffers=False)
        model_without_ddp = model.module

    for n , p in model.named_parameters():
        # if ("text_encoder" in n or "input_proj" in n or "resizer" in n or 'backbone' in n

        # ):
        # if  ('resoluter' in  n or "transformer.weight" in  n or "decoder" in n  ):
        # if ('r_q' in n or 'decoder' in  n or 'temp' in n  or "transformer.encoder" in n):
        # if ("retriver"  in n or "rcoder" in n ):
        #     pass
        # else:            src = torch.cat([src,knowl_memory],
        #                              dim=0)  # [HW+batch_max_sequence_len,bs,d_model]
        #             mask = torch.cat([mask, knowl_mask],
        #                               dim=1)  # [bs,HW+batch_max_sequence_len]
        #             pos_embed = torch.cat([pos_embed,torch.zeros_like(knowl_memory),
        #                                     ], dim=0)
        #             src, eatt = self.encoder.layers[0](src, src_key_padding_mask=mask,
        #                                               pos=pos_embed)
        #             src=src[:len_src]
        #             pos_embed =pos_embed[:len_src]
        #             mask = mask[:,:len_src]
        #
        #
        #
        #             refer_memory, teatt = self.resloter.layers[0](refer_memory, knowl_memory,src[len_src:],
        #
        #                                                    tgt_key_padding_mask=refer_mask,
        #                                                    memory_key_padding_mask=knowl_mask,
        #                                                    pos=torch.zeros_like(knowl_memory))
        #
        #             knowl_memory= src[len_src:]
        #
        #
        #             src = torch.cat([src, refer_memory],
        #                             dim=0)  # [HW+batch_max_sequence_len,bs,d_model]
        #             mask = torch.cat([mask, refer_mask],
        #                              dim=1)  # [bs,HW+batch_max_sequence_len]
        #             pos_embed = torch.cat([pos_embed, torch.zeros_like(refer_mask),
        #                                    ], dim=0)
        #             src, eatt = self.encoder.layers[0](src, src_key_padding_mask=mask,
        #                                                pos=pos_embed)
        #             src = src[:len_src]
        #             pos_embed = pos_embed[:len_src]
        #             mask = mask[:, :len_src]
        #     p.requires_grad =False
        # if (("encoder" in n and ('3' in n or '4' in n and '5'  in n)) or 'decoder' in n ):
        #     pass
        # else:
        #     p.requires_grad = False
        # if (  "backbone" in n ):
        #     p.requires_grad = False
        if (  "backbone" in n):
            p.requires_grad = False
        # for layer in range(12):
            # if str(layer) in n and layer  >3 and "transformer.encoder" in n:
            #     p.requires_grad = text_memory
            # if str(layer) in n and layer  >3 and "transformer.decoder" in n:
            #     p.requires_grad = False
            # if str(layer) in n and layer  <3 and "text_encoder" in n:
            #     p.requires_grad = False
        # if (  "backbone" in n or"transformer.encoder" in n  ):
        #     p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)


    ## Set up optimizers，设置优化器
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr": args.text_encoder_lr,
        },
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")


    ## 获取训练数据集
    # Train dataset
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)
    dataset_test = build_dataset('test', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True,drop_last=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_train = torch.utils.data.SequentialSampler(dataset_train)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.num_workers)




    ##用于从模型的检查点恢复训练
    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"],strict=False)
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
    elif args.load:
        checkpoint = torch.load(args.load, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"],strict=False)
        # model_without_ddp.transformer.resloter.load_state_dict(
        #     model_without_ddp.transformer.encoder.state_dict(),strict=False)
        # model_without_ddp.transformer.kv_encoder.load_state_dict(
        #     model_without_ddp.transformer.encoder.state_dict(), strict=False)

        # old_embedding = model_without_ddp.transformer.text_encoder.get_input_embeddings()
        # # 创建新的嵌入层
        # new_embedding = nn.Embedding(old_embedding.num_embeddings + 2103, old_embedding.embedding_dim).to(old_embedding.weight.device)
        # # 初始化新的嵌入层权重，保留原始权重
        # new_embedding.weight.data[:old_embedding.num_embeddings, :] = old_embedding.weight.data
        # new_embedding.weight.data[old_embedding.num_embeddings:, :] = torch.rand(2103,
        #                                                                          old_embedding.embedding_dim)
        # model_without_ddp.transformer.text_encoder.set_input_embeddings(new_embedding)

    ##  测试的代码


    ##  测试的代码
    # Runs only evaluation, by default on the validation set unless --test is passed.
    if args.eval:
        test_stats = evaluate(
            model=model,
            criterion=criterion,
            weight_dict=weight_dict,
            data_loader=data_loader_test,
            device=device,
            args=args,
            )

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)
        return

    else:

        ## 训练的代码
        # Runs training and evaluates after every --eval_skip epochs
        print("Start training")
        start_time = time.time()
        best_metric = 0.0
        for epoch in range(args.start_epoch, args.epochs):

            print(f"Starting epoch {epoch}")
            if args.distributed:
                sampler_train.set_epoch(epoch)

            #进行一个轮次的训练
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=data_loader_train,
                weight_dict=weight_dict,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                args=args,
                max_norm=args.clip_max_norm,
            )
            ## 如果设置了输出路径，则将训练权重文件缓存到输出路径中
            if args.output_dir:
                checkpoint_paths = [output_dir / "checkpoint.pth"]
                # extra checkpoint before LR drop and every 2 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                    checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
            ## 隔几个epoch就验证以下
            if epoch % args.eval_skip == 0:
                eval_stats = {}

                curr_test_stats = evaluate(
                    model=model,
                    criterion=criterion,
                    weight_dict=weight_dict,
                    data_loader=data_loader_val,
                    device=device,
                    args=args,
                )
                eval_stats.update({  k: v for k, v in curr_test_stats.items()})


            ## 记录一些状态并加载到log.txt文件中
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in eval_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if dist.is_main_process()   :
                writer.add_scalar('train_loss',log_stats['train_loss'],epoch)
                writer.add_scalar('test_loss',log_stats['test_loss'],epoch)
                writer.add_scalar('test_accu',log_stats['test_accu'],epoch)

            if args.output_dir and dist.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            ##保存验证结果最优异的模型到输出路径中
            if epoch % args.eval_skip == 0:
                torch.cuda.empty_cache()
                metric = np.mean([v for k, v in eval_stats.items() if "acc" in k])

                if args.output_dir and metric > best_metric:
                    best_metric = metric
                    checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                    # extra checkpoint before LR drop and every 100 epochs
                    for checkpoint_path in checkpoint_paths:
                        dist.save_on_master(
                            {
                                "model": model_without_ddp.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "args": args,
                            },
                            checkpoint_path,
                        )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()],)
    args = parser.parse_args()
    ## 如果输出路径存在，则创建该路径
    if args.output_dir:
        #parents参数是父目录不存在，是否创建父目录、exist_ok是只有在目录不存在时才创建，目录存在不会抛出异常
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)