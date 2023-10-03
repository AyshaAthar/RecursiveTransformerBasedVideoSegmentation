import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torch.nn.functional as FF

import math
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
from einops import rearrange
import wandb
import gc


from segm_video.utils.logger import MetricLogger
from segm_video.metrics import gather_data, compute_metrics
from segm_video.model import utils
from segm_video.data.utils import seg_to_rgb, rgb_denormalize, IGNORE_LABEL
import segm_video.utils.torch as ptu
from segm_video.model.utils import num_params,inference

def train_one_epoch(
        dataset,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        epoch,
        amp_autocast,
        loss_scaler,
    ):
        """ Train one epoch """
        wandb_log = 0
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.CrossEntropyLoss(ignore_index=255)
        logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"
        print_freq = 100

        model.train()
        data_loader.set_epoch(epoch)
        num_updates = epoch * len(data_loader)
        loss_list=[]
        total_loss = 0
        for batch in logger.log_every(data_loader, print_freq, header):
            wandb_log = wandb_log + 1
            im = batch["im"]
            im = im.to(ptu.device)

            seg_gt = batch["segmentation"].long()

            seg_gt = seg_gt.to(ptu.device).long()

            with amp_autocast():
                pred_seg, vit_embds, predictor_embds, corrector_embds = model(im)
                corrector_embds = corrector_embds.to(ptu.device)
                predictor_embds = predictor_embds.to(ptu.device)
                vit_embds = vit_embds.to(ptu.device)

                loss1 = criterion1(predictor_embds,vit_embds[:, 1:]) 

                pred_seg = pred_seg.to(ptu.device)
                pred_seg = rearrange(pred_seg, "b t n h w -> b n t h w")
                if dataset=="cityscapesseq":
                    ann_filename = batch["ori_filename"]
                    seq = (batch['sequence_filename'])
                    index = seq.index(ann_filename)
                    pred_seg_annotated = pred_seg[:,:,index:index+1,:,:]
                    pred_seg_annotated = pred_seg_annotated.to(ptu.device)
                    loss2 = criterion2(pred_seg_annotated,seg_gt) 
                else:
                    loss2 = criterion2(pred_seg,seg_gt) 

                combined_loss = loss2 + 0.001*loss1

                if(wandb_log % 100):
                    wandb.log({"Cross Entropy Loss": loss2.item(), "MSE loss": loss1.item()})

            loss_list.append(combined_loss.item())

            if not math.isfinite(combined_loss.item()):
                print("Loss is {}, stopping training".format(loss2.item()))
                sys.exit(1)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)
            torch.cuda.synchronize(device = ptu.device)
            logger.update(
                final_loss=combined_loss.item(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )     
        mean_loss = np.mean(loss_list)
        print(" Train loss", mean_loss)
        wandb.log({"Train loss": mean_loss})
        return logger

@torch.no_grad()
def evaluate(
    dataset,
    model,
    data_loader,
    window_size,
    window_stride,
    amp_autocast
):
    """ Evaluating model """
    if(dataset=="cityscspesseq"):
        n_cls=19
    else:
        n_cls=23
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    criterion1 = torch.nn.MSELoss()

    criterion2 = torch.nn.CrossEntropyLoss(ignore_index=255)   

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    loss_list = []
    seg_pred_maps = {}
    seg_gt_maps = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = batch["im"][0]
        segs = batch["segs"][0]

        ims = ims.type(torch.FloatTensor).to(ptu.device)
        if dataset=="cityscapesseq":
            segs = rearrange(segs, "(b t) h w -> b t h w", t=1)
            segs = (segs).to(ptu.device).long()
        else:
            segs = torch.stack(segs)
            segs = rearrange(segs,"t b h w -> b t h w")
            segs = segs.long().to(ptu.device)

        

        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = torch.cat(ori_shape, dim=0)[0:2].numpy()
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred, predictor_embds, corrector_embds, vit_embds = inference(model,ims,ims_metas,ori_shape, window_size,window_stride,batch_size=1,)
            #loss1 = criterion1(predictor_embds,vit_embds[:, 1:])
            if dataset!="cityscapesseq":
                seg_pred = seg_pred.reshape(1,4,23,135,240)

            seg_pred = rearrange(seg_pred,"b t n h w -> b n t h w")
            seg_pred = seg_pred.to(ptu.device)
            seg_pred = seg_pred.float()
            if dataset=="cityscapesseq":
                ann_filename = batch["im_metas"][0]["ori_filename"]
                seq = (batch['im_metas'][0]['sequence_filename'])
                index = seq.index(ann_filename)
                pred_seg_annotated = seg_pred[:,:,index:index+1,:,:]
                loss2 = criterion2(pred_seg_annotated,segs)
            else:
                loss2 = criterion2(seg_pred,segs)

            #loss_list.append(loss2.item() + 0.001*loss1.item())
            loss_list.append(loss2.item())

            seg_pred = rearrange(seg_pred,"b n t h w -> b t n h w")

            seg_pred = FF.softmax(seg_pred, 2)

            seg_pred = seg_pred.argmax(2)
            
            if(dataset!="cityscapesseq"):
                seg_pred_maps[filename] = seg_pred.detach().cpu()
                seg_gt_maps[filename] = segs.detach().cpu()
            else:
                seg_gt_maps[filename] = segs.squeeze(dim=0).detach().cpu()
                seg_pred_maps[filename] = seg_pred.squeeze().detach().cpu()[index:index+1,:,:]


    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )
    for k, v in scores.items():
        if(k=="cat_iou"):
            continue
        else:
            logger.update(**{f"{k}": v, "n": 1})
    wandb.log({"test_epoch_loss": np.mean(loss_list)})
    
    visualize_results(model, dataset, data_loader, window_size, window_stride, ptu.device)
    
    print("Test loss mean", np.mean(loss_list))

    del seg_pred_maps
    del seg_gt_maps
    del scores
    gc.collect()

    return logger

def show(grids):
    fig, axs = plt.subplots(nrows=len(grids), squeeze=False)
    fig.set_size_inches(25,8)

    for i, grid in enumerate(grids):
        grid = grid.detach()
        grid = F.to_pil_image(grid)

        axs[i, 0].imshow(np.asarray(grid))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.close('all')
    wandb.log({"outputs" : wandb.Image(fig)}) 

def visualize_results(model, dataset, test_loader,window_size,window_stride, device):
    batch = next(iter(test_loader))

    ims = batch["im"][0]
    segs = batch["segs"][0]
    b, t, n, h, w = ims.shape
    
    ims = ims.type(torch.FloatTensor).to(ptu.device)
    
    if dataset=="cityscapesseq":
        segs = rearrange(segs, "(b t) h w -> b t h w", t=1)
    else:
        segs = torch.stack(segs)
        segs = rearrange(segs,"t b h w -> b t h w")

    segs = (segs).to(ptu.device).long()

    colors= batch["colors"]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = torch.cat(ori_shape, dim=0)[0:2].numpy()

    visual_grid =[]

    model.eval() 
    with torch.no_grad():

        seg_pred, predictor_embds, corrector_embds, vit_embds = inference(model,ims,ims_metas,ori_shape,window_size,window_stride,1)
        if dataset!="cityscapesseq":
            seg_pred = seg_pred.reshape(1,4,23,135,240)

        seg_pred = FF.softmax(seg_pred, 2)

        seg_pred = seg_pred.to(ptu.device)
        seg_pred = seg_pred.float()

        pred_rgb = seg_pred.argmax(2)

        pred_rgb = seg_to_rgb(pred_rgb, colors)
        pred_rgb = torch.stack(pred_rgb)
        pred_rgb_uint = (255 * pred_rgb.cpu().numpy()).astype(np.uint8)        

        seg_rgb = seg_to_rgb(segs, colors)
        seg_rgb = torch.stack(seg_rgb)
        seg_rgb_uint = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)        

        seg_rgb_list=[]
        pred_rgb_list=[]
        if dataset=="cityscapesseq":
            seg_rgb_list.append(torch.tensor(seg_rgb_uint[0][0]).permute(2,0,1))
            for i in range(t):
                pred_rgb_list.append(torch.tensor(pred_rgb_uint[i][0]).permute(2,0,1))
        else:
            for i in range(t):
                seg_rgb_list.append(torch.tensor(seg_rgb_uint[i][0]).permute(2,0,1))
                pred_rgb_list.append(torch.tensor(pred_rgb_uint[i][0]).permute(2,0,1))

        visual_grid.append(make_grid(seg_rgb_list, padding=5))
        visual_grid.append(make_grid(pred_rgb_list, padding=5))

    show(visual_grid)