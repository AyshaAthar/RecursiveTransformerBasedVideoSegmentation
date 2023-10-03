import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import math
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path
import yaml
import json
import os
import argparse
from contextlib import suppress
import wandb
import sys
import click

from timm.utils import NativeScaler

from segm_video.model.vivit.video_vit import ViViT
from segm_video.utils import distributed
import segm_video.utils.torch as ptu
from segm_video import config
from segm_video.data.factory import create_dataset
from segm_video.optim.factory import create_optimizer, create_scheduler
from segm_video.utils.distributed import sync_model
from segm_video.model.utils import checkpoint_filter_fn
from engine import train_one_epoch, evaluate

@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--pretrain-log-dir", type=str, help="Pre-train directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
def main(log_dir,
    pretrain_log_dir,
    dataset,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
):
    #WANDB INITIALISATION
    wandb.init(project="AIS-thesis", entity="aysha_athar", dir="../")

    # start distributed mode
    ptu.set_gpu_mode(True)

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]

    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]
    if(dataset=="cityscapesseq"):
        optimizer="Adam"
        
    # model config
    im_size = dataset_cfg["im_size"]
    crop_size = dataset_cfg.get("crop_size", im_size)
    window_size = dataset_cfg.get("window_size", im_size)
    window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path

    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]

    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size

    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-6,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path("../"+log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth" 

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]
  
    train_loader = create_dataset(dataset_kwargs)

    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False

    val_loader = create_dataset(val_kwargs)

    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    patch_size = net_kwargs['patch_size']
    d_model = net_kwargs["d_model"]
    n_layers = net_kwargs["n_layers"]
    n_layers_pred = net_kwargs["n_layers_pred"]
    n_layers_corr = net_kwargs["n_layers_corr"]
    n_heads = net_kwargs["n_heads"]

    #Model initialisation
    model = ViViT((crop_size,crop_size), patch_size, n_layers , n_layers_pred, n_layers_corr, d_model, 4*d_model, n_heads, n_cls, decoder)
    
    #Loading weights for pretraining
    path = os.path.expandvars("../"+pretrain_log_dir + "/checkpoint.pth")
    state_dict = torch.load(path, map_location="cpu")
    
    filtered_dict = checkpoint_filter_fn(state_dict, model.vit_embs.encoder)

    model.decoder.load_state_dict(filtered_dict,strict = False)
    model.vit_embs.load_state_dict(filtered_dict,strict = False)
    
    #Freezing backbone
    for param in model.vit_embs.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params",total_params)

    model = model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v

    if(optimizer=="sgd"):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr ,momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

    lr_scheduler = create_scheduler(opt_args, optimizer)
    amp_autocast = suppress
    loss_scaler = None

    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
    else:
        sync_model(log_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    print(f"Train dataset length: {len(train_loader.dataset)}")

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            dataset,
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
        )
        #Evaluate epoch
        eval_logger = evaluate(
        dataset,
        model,
        val_loader,
        window_size,
        window_stride,
        amp_autocast)

        print(f"Stats [{epoch}]:", eval_logger, flush=True)
        print("")

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            torch.save(snapshot, checkpoint_path)

        print(f"Stats [{epoch}]:", eval_logger, flush=True)
        print("")

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}

            val_stats = {
                k: meter.global_avg for k, meter in eval_logger.meters.items()
            }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader)
            }
            wandb.log(log_stats)
            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == "__main__":
    main()