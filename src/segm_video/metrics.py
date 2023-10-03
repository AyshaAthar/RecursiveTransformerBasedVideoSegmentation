import torch
import numpy as np
import torch.distributed as dist
import segm_video.utils.torch as ptu

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
import torchvision.transforms as T
import mmcv


"""
ImageNet classifcation accuracy
"""
def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""

def gather_data(seg_pred, tmp_dir=None):
    """
    distributed data gathering
    prediction and ground truth are stored in a common tmp directory
    and loaded on the master node to compute metrics
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("$DATASET/temp")
    else:
        tmpprefix = os.path.expandvars(tmp_dir)
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device=ptu.device)
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    #dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    """
    Save results in temp file and load them on main process
    """
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    pkl.dump(seg_pred, open(tmp_file, "wb"))
    #dist.barrier()
    seg_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            part_seg_pred = pkl.load(open(tmpdir / f"part_{i}.pkl", "rb"))
            seg_pred.update(part_seg_pred)
        shutil.rmtree(tmpdir)
    return seg_pred


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    tmp_dir=None,
    distributed=False,
):
    ret_metrics_mean = torch.zeros(3, dtype=float, device=ptu.device)
    if ptu.dist_rank == 0:
        list_seg_pred = []
        list_seg_gt = []
        keys = sorted(seg_pred.keys())
        for k in keys:
            list_seg_pred.append(np.asarray(seg_pred[k]))
            list_seg_gt.append(np.asarray(seg_gt[k]))
        ret_metrics = mean_iou(
            results=list_seg_pred,
            gt_seg_maps=list_seg_gt,
            num_classes=n_cls,
            ignore_index=ignore_index,
        )

        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=ptu.device,
        )
        cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    if ret_cat_iou and ptu.dist_rank == 0:
        ret["cat_iou"] = cat_iou
    del seg_pred, seg_gt, list_seg_pred, list_seg_gt
    return ret


## Below functions adapted from https://github.com/wanghao9610/TMANet 
def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    mask = (label != ignore_index)

    pred_label = torch.from_numpy(pred_label)

    pred_label = pred_label.numpy()

    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect
    del pred_label, label

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=float)
    total_area_label = np.zeros((num_classes, ), dtype=float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    del results, gt_seg_maps
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """
    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num)

    del results, gt_seg_maps
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category dice, shape (num_classes, )
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category evalution metrics, shape (num_classes, )
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes,
                                                     ignore_index=ignore_index)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    
    acc = total_area_intersect / total_area_label   # TP / (TP + FN)
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    del gt_seg_maps, results
    return ret_metrics