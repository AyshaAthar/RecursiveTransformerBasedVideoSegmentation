import os.path as osp

import torch
import numpy as np
from torch.utils.data import Dataset
import mmcv
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmcv.utils import print_log
from mmseg.datasets.pipelines import Compose
from mmseg.datasets import PIPELINES
from os import listdir
from pathlib import Path
import os
import random
from mmcv.utils import deprecated_api_warning

from matplotlib import pyplot as plt

from PIL import Image

from mmseg.datasets.custom import CustomDataset
from pathlib import Path
from segm_video.data.base import BaseMMSeg
from segm_video.data import utils
from segm_video.config import dataset_dir

import collections
from collections.abc import Sequence  
from mmcv.parallel import DataContainer as DC 

@DATASETS.register_module()
class CustomVideoDataset(Dataset):
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 sequence_dir=None,
                 sequence_suffix='.png',
                 sequence_num=1,
                 random_select=False,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.sequence_dir = sequence_dir
        self.sequence_suffix = sequence_suffix
        self.sequence_num=4
        
        self.annotated_img = 19
        
        if(self.sequence_num<10):
            self.sequence_start = self.annotated_img - (self.sequence_num-1)
            self.sequence_stop = self.annotated_img -1
        else:
            self.sequence_start = abs(self.annotated_img - (self.sequence_num-1))
            self.sequence_stop = 30 - self.sequence_num
            if(self.sequence_stop<self.sequence_start):
                self.sequence_start=0
            
        if sequence_num > 1 and random_select:
            self.random_select = True
        else:
            self.random_select = False
        self.split=split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.sequence_dir is None or osp.isabs(self.sequence_dir)):
                self.sequence_dir = osp.join(self.data_root, self.sequence_dir)

        
        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir, self.seg_map_suffix,
                                               self.sequence_dir, self.sequence_suffix,
                                               self.split
                                               )
        
    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, sequence_dir,
                         sequence_suffix, split):
        img_infos = []

        for img in sorted(mmcv.scandir(img_dir, img_suffix, recursive=True)):
            path = Path(sequence_dir+'/'+img.split('/')[0])
            path = listdir(path)
            path.sort()
            indexofimg = path.index(img.split('/')[1])
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                img_info['colors'] = self.PALETTE
                img_info['ann_filename'] = img
            if sequence_dir is not None:
                if(split=='train'):
                    frames = []
                    frame_index = 16
                    finalstart = indexofimg - 19 + frame_index
                    finalstop = finalstart + self.sequence_num
                    for index in range(finalstart , finalstop):
                        frame = path[index]
                        frames.append(img.split('/')[0]+'/'+frame)

                elif(split=='test'or split=="val"):
                    
                    frames = []
                    if(self.sequence_num<=19):
                        finalstart = indexofimg - self.sequence_num + 1 
                        finalstop = indexofimg + 1
                    else:
                        finalstart = indexofimg - 19
                        #finalstop = finalstart + self.sequence_num + 1 , commented on 21.09
                        finalstop = finalstart + self.sequence_num 

                    for index in range(finalstart , finalstop):
                        frame = path[index]
                        frames.append(img.split('/')[0]+'/'+frame)

                img_info['sequence'] = dict(frames=frames)
            img_infos.append(img_info)
            #if(len(img_infos)==1):
                #break;

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_sequence_info(self, idx):
        """Get sequence by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Sequence image info of specified index.
        """

        return self.img_infos[idx]['sequence']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['sequence_prefix'] = self.sequence_dir
        if self.custom_classes:
            results['label_map'] = self.label_map
        results['colors'] = self.PALETTE

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        sequence_info = self.get_sequence_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, sequence_info=sequence_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        sequence_info = self.get_sequence_info(idx)
        results = dict(img_info=img_info, sequence_info=sequence_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette
    
    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = mmcv.imresize(gt_seg_map, (768,768), interpolation='nearest') 

            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps
    
    
    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette


# In[2]:


@DATASETS.register_module()
class cityscapes_new(CustomVideoDataset):
    """Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):

        super().__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            sequence_suffix='_leftImg8bit.png',
            random_select=True,
            **kwargs)
        
        
CITYSCAPES_CONFIG_PATH = "./segm_video/data/config/cityscapesseq.py"

CITYSCAPES_CATS_PATH = "./segm_video/data/config/cityscapesseq.yml"
from segm_video.data.base import BaseMMSeg
class CityscapesseqVP(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(image_size, crop_size, split, CITYSCAPES_CONFIG_PATH,**kwargs)
        self.names, self.colors = utils.dataset_cat_description(CITYSCAPES_CATS_PATH)
        self.n_cls = 19
        self.ignore_label = 255
        self.reduce_zero_label = False

    def update_default_config(self, config):

        root_dir = dataset_dir()
        path = Path(root_dir) / "cityscapes_new"
        config.data_root = path

        config.data[self.split]["data_root"] = path
        config = super().update_default_config(config)

        return config

    def test_post_process(self, labels):
        labels_copy = np.copy(labels)
        cats = np.unique(labels_copy)
        for cat in cats:
            labels_copy[labels == cat] = CSLabels.trainId2label[cat].id
        return labels_copy

