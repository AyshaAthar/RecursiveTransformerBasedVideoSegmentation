import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import numpy as np
from pathlib import Path
import os.path as osp

from PIL import Image, ImageOps, ImageFilter

from mmseg.datasets import build_dataset
import mmcv
from mmcv.utils import Config
import sys

from segm_video.data.utils import STATS, IGNORE_LABEL
from segm_video.data import utils

class BaseMMSeg(Dataset):
    def __init__(
        self,
        image_size,
        crop_size,
        split,
        config_path,
        normalization,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.normalization = STATS[normalization].copy()
        self.ignore_label = None
        for k, v in self.normalization.items():
            v = np.round(255 * np.array(v), 2)
            self.normalization[k] = tuple(v)
        print(f"Use normalization: {self.normalization}")
        
        if "cityscapes" in config_path:
            self.dataset_type = "cityscapes"
        else:
            self.dataset_type = "synpick"
        if(split=="val"):
            test_mode = True
        config = Config.fromfile(config_path)  

        self.ratio = config.max_ratio
        self.dataset = None
        self.config = self.update_default_config(config)
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def update_default_config(self, config):
        train_splits = ["train", "trainval"]
        if self.split in train_splits:
            config_pipeline = getattr(config, f"train_pipeline")
        else:
            config_pipeline = getattr(config, f"{self.split}_pipeline")
            
        img_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug1"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, op in enumerate(config_pipeline):
            op_type = op["type"]
            if op_type == "Resize1":
                op["img_scale"] = img_scale
            elif op_type == "RandomCrop1":
                op["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif op_type == "Normalize1":
                op["mean"] = self.normalization["mean"]
                op["std"] = self.normalization["std"]
            elif op_type == "Pad1":
                op["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = op
        if self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = img_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def set_multiscale_mode(self):
        self.config.data.val.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.val.pipeline[1]["flip"] = True
        self.config.data.test.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.test.pipeline[1]["flip"] = True
        self.dataset = build_dataset(getattr(self.config.data, f"{self.split}"))

    def __getitem__(self, idx):
        data = self.dataset[idx]
        train_splits = ["train", "trainval"]

        if self.split in train_splits:
            im = data["sequence_imgs"].data
            seg = data["gt_semantic_seg"].data.squeeze(1)
        else:
            im = [im.data for im in data["sequence_imgs"]]
            seg = None

        out = dict(im=im)
        if self.split in train_splits:
            if(self.dataset_type == "cityscapes"):
                out["ori_filename"] = data["img_metas"]["ori_filename"]
                out["sequence_filename"] = data["img_metas"]["sequence_filename"]
            else:
                seg[seg==255]=22
                out["ori_filename"] = data["img_metas"]["ori_filename"]
                out["sequence_filename"] = data["img_metas"]["sequence_filename"]
            out["segmentation"] = seg
        else:
            for meta in data["img_metas"]:
                info = meta
            im_metas = [meta for meta in data["img_metas"]]
            out["im_metas"] = im_metas
            out["colors"] = self.colors
            
            gt_seg_maps,filenames = self.get_gt_seg_maps(idx)
            out["segs"] = gt_seg_maps
            out["filenames"] = filenames

        return out

    def get_gt_seg_maps(self,idx):
        dataset = self.dataset[idx]
        ann_dir = dataset['img_metas'][0]['ann_dir']
        gt_seg_maps = []
        filenames=[]
        if self.dataset_type == "cityscapes":
            for img_info in dataset['img_metas']:
                seg_map = osp.join(ann_dir, img_info['ori_filename'].replace('_leftImg8bit.png','_gtFine_labelTrainIds.png'))

                gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
                gt_seg_map[gt_seg_map == self.ignore_label] = IGNORE_LABEL
                if self.reduce_zero_label:
                    gt_seg_map[gt_seg_map != IGNORE_LABEL] -= 1
                #gt_seg_maps[img_info["filename"]] = gt_seg_map
                gt_seg_maps.append(gt_seg_map)
                filenames.append(img_info['filename'])
        else:
            for img_info in dataset['img_metas']:
                maps = []
                filename = []
                seg_maps = img_info['sequence_filename']
                for f in seg_maps:
                    seg_map = osp.join(ann_dir, f.replace('.jpg','.png'))
                    gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
                    if self.reduce_zero_label:
                        # avoid using underflow conversion
                        gt_seg_map[gt_seg_map != IGNORE_LABEL] -= 1
                    gt_seg_map[gt_seg_map == 255] = 22 
                    
                    maps.append(gt_seg_map)
                    filename.append(f)
                             
                gt_seg_maps.append(maps)
                filenames.append(filename)
            
        return gt_seg_maps,filenames

    def __len__(self):
        return len(self.dataset)

    @property
    def unwrapped(self):
        return self

    def set_epoch(self, epoch):
        pass

    def get_diagnostics(self, logger):
        pass

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return
