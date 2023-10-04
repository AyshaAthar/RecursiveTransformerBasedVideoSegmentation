import sys
import gc
import click
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import shutil
import os
from einops import rearrange

import torch
import torch.nn.functional as FF
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from segm_video.utils import distributed
from segm_video.utils.logger import MetricLogger
import segm_video.utils.torch as ptu

from segm_video.data.factory import create_dataset
from segm_video.metrics import gather_data, compute_metrics

from segm_video.model.utils import inference
from segm_video.data.utils import seg_to_rgb, rgb_denormalize, IGNORE_LABEL
from segm_video import config
            
from segm_video.model.utils import inference
from segm_video.model.utils import checkpoint_filter_fn

from segm_video.model.vivit.video_vit import ViViT

def process_batch(
    model, dataset, batch, window_size, window_stride, window_batch_size,
):
    """ Process each batch """
    ims = batch["im"][0]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = torch.cat(ori_shape, dim=0)[0:2].numpy()
    filename = batch["im_metas"][0]["ori_filename"][0]
    ims = ims.type(torch.FloatTensor).to(ptu.device)
    
    segs = batch["segs"][0]
    
    if dataset=="cityscapesseq":
            segs = rearrange(segs, "(b t) h w -> b t h w", t=1)
            segs = (segs).to(ptu.device).long()
    else:
        segs = torch.stack(segs)
        segs = rearrange(segs,"t b h w -> b t h w")
        segs = segs.long().to(ptu.device)

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    seg_pred,vit_embds, predictor_embds, corrector_embds = inference(model,ims,ims_metas,ori_shape,window_size,window_stride,1) 
    seg_pred = seg_pred.argmax(2)
    
    ann_filename = batch["im_metas"][0]["ori_filename"]
    seq = (batch['im_metas'][0]['sequence_filename'])
    
    if(dataset=="cityscapesseq"):
        pred_seg_annotated = seg_pred[:,3:4,:,:]
    else:
        pred_seg_annotated = seg_pred
        
    return filename, pred_seg_annotated.squeeze().cpu(), segs.squeeze(dim=0).detach().cpu()


def eval_dataset(
    model,
    multiscale,
    model_dir,
    window_size,
    window_stride,
    window_batch_size,
    frac_dataset,
    dataset_kwargs,
):
    """ Function to evaluate the dataset on the given model, computes miou and accuracy """
    db = create_dataset(dataset_kwargs)
    normalization = db.dataset.normalization
    dataset_name = dataset_kwargs["dataset"]
    im_size = dataset_kwargs["image_size"]
    cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls
    
    if multiscale:
        db.dataset.set_multiscale_mode()

    logger = MetricLogger(delimiter="  ")
    header = ""
    print_freq = 50
    
    idx = 0
    seg_pred_maps = {}
    seg_gt_maps = {}
    for batch in logger.log_every(db, print_freq, header):

        colors = batch["colors"]
        filename, seg_pred, seg_gt = process_batch(
            model, dataset_name, batch, window_size, window_stride, window_batch_size,
        )
        seg_pred_maps[filename] = seg_pred
        seg_gt_maps[filename] = seg_gt

        idx += 1
        if idx > len(db) * frac_dataset:
            break

    if ptu.distributed:
        torch.distributed.barrier()
        seg_pred_maps = gather_data(seg_pred_maps)

    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )
    print(scores)
    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not multiscale else "multi_scale"
        suffix = "ss" if not multiscale else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou" and k != "inference":
                scores[k] = v.item()
            if k != "cat_iou":
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open("../" + model_path / f"scores_{suffix}.yml", "w") as f:
            f.write(scores_str)
    del seg_pred_maps
    del seg_gt_maps 
    gc.collect()
    
    
def load_model(model_path, dataset_name):
    """ Function to load the model """
    
    variant_path = "../" + model_path + "/variant.yml"
    print(variant_path)
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]
    
    crop_size = net_kwargs["image_size"]
    patch_size = net_kwargs["patch_size"]
    n_layers = net_kwargs["n_layers"]
    n_layers_pred = net_kwargs["n_layers_pred"]
    n_layers_corr = net_kwargs["n_layers_corr"]
    d_model = net_kwargs["d_model"]
    n_heads = net_kwargs["n_heads"]
    decoder = net_kwargs["decoder"]["name"]
    
    if(dataset_name == "cityscapesseq"):
        n_cls=19
    else:
        n_cls=23
        
    model = ViViT((crop_size), patch_size, n_layers , n_layers_pred, n_layers_corr, d_model, 4*d_model, n_heads, n_cls, decoder)
    path = os.path.expandvars("../" + model_path + "/checkpoint.pth")

    #Loading weights
    state_dict = torch.load(path, map_location="cpu")
    filtered_dict = checkpoint_filter_fn(state_dict, model.vit_embs.encoder)
    model.load_state_dict(filtered_dict,strict = False)

    return model, variant


@click.command(help="")
@click.option("--model-path", type=str, help="Path to model")
@click.option("--dataset-name", type=str, help="Dataset name to evalaute")
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=1, type=int)
@click.option("-frac-dataset", "--frac-dataset", default=1.0, type=float)
def main(
    model_path,
    dataset_name,
    im_size,
    multiscale,
    window_size,
    window_stride,
    window_batch_size,
    frac_dataset,
):

    model_dir = Path(model_path).parent

    # start distributed mode
    ptu.set_gpu_mode(True)

    model, variant = load_model(model_path, dataset_name)
    patch_size = model.patch_size
    model.to(ptu.device)
    model.eval()
    
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][dataset_name]
    normalization = variant["dataset_kwargs"]["normalization"]
    if im_size is None:
        im_size = dataset_cfg.get("im_size", variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = variant["inference_kwargs"]["window_size"]
    if window_stride is None:
        window_stride = variant["inference_kwargs"]["window_stride"] - 32

    dataset_kwargs = dict(
        dataset=dataset_name,
        image_size=im_size,
        crop_size=im_size,
        patch_size=patch_size,
        batch_size=1,
        num_workers=10,
        split="val",
        normalization=normalization,
        crop=False,
        rep_aug=False,
    )

    eval_dataset(
        model,
        multiscale,
        model_path,
        window_size,
        window_stride,
        window_batch_size,
        frac_dataset,
        dataset_kwargs,
    )

    sys.exit(1)

if __name__ == "__main__":
    main()
