#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
SynpickVP dataset: Obtaining video sequences from the SynpickVP dataset, along with
the corresponding semantic segmentation maps.
"""

import json
import math
import os
from tqdm import tqdm
import imageio
import torch
from torchvision import transforms
import numpy as np
from collections import defaultdict

from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T


class SynpickVP():
    """
    Each sequence depicts a robotic suction cap gripper that moves around in a red bin filled with objects.
    Over the course of the sequence, the robot approaches 4 waypoints that are randomly chosen from the 4 corners.
    On its way, the robot is pushing around the objects in the bin.
    Args:
    -----
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    seq_step: int
        Temporal resolution at which we use frames. seq_step=2 means we use one frame out of each two
    max_overlap: float
        Determins amount of overlap between consecuitve sequences, given as percentage of num_frames.
        For instance, 0.25 means that consecutive sequence will overlap for (1 - 0.25)=75% of the frames
    img_size: tuple
        Images are resized to this resolution
    """

    DATA_PATH = "/home/nfs/inf6/data/datasets/SynpickRaw/processed"

    CATEGORIES = ["background","master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
                  "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
                  "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
                  "large_clamp", "extra_large_clamp", "foam_brick", "gripper"]
    NUM_CLASSES = len(CATEGORIES)  # 22
    print(NUM_CLASSES)

    BIN_SIZE = (373, 615)
    SKIP_FIRST_N = 72            # To skip the first few frames in which gripper is not visible.
    GRIPPER_VALID_OFFSET = 0.01  # To skip sequences where the gripper_pos is near the edges of the bin.

    NICE_SIZES = [(64, 112), (136, 240)]
    NUM_FRAMES_LIMITS = {
            "train": [10, 50],
            "val": [50, 50],
            "test": [50, 50],  # please keep this fixed
        }

    def __init__(self, split, num_frames, seq_step=2, max_overlap=0.25, img_size=(135, 240)):
        """
        Dataset initializer
        """
        assert split in ["train", "val", "test"]
        assert max_overlap <= 0.95 and max_overlap >= 0
        self.data_dir = os.path.join(SynpickVP.DATA_PATH, split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Synpick dataset does not exist in {self.data_dir}...")
        print(f"Loading SypickVP {split} set...")
        num_frames = 1

        self.split = split
        self.img_size = img_size
        self.num_frames = num_frames
        self.seq_step = seq_step
        self.max_overlap = max_overlap
        
        ###NEW
        self.n_cls=23

        # obtaining paths to data
        images_dir = os.path.join(self.data_dir, "rgb")
        scene_dir = os.path.join(self.data_dir, "scene_gt")
        masks_dir = os.path.join(self.data_dir, "masks")
        instances_dir = os.path.join(self.data_dir, "instance_masks")
        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.scene_gt_fps = [os.path.join(scene_dir, scene) for scene in sorted(os.listdir(scene_dir))]
        self.mask_fps = [os.path.join(masks_dir, mask_fp) for mask_fp in sorted(os.listdir(masks_dir))]
        self.instance_fps = [os.path.join(instances_dir, inst_fp) for inst_fp in sorted(os.listdir(instances_dir))]

        # obtaining pose information from the different objectd
        self.object_poses = {}
        self._create_frame_to_obj_kpoints_dict()

        # generating sequences
        self.valid_idx = []
        self.index=[]
        self.allow_seq_overlap = (split == "train")
        
        #self.valid_ids = sorted(os.listdir("/home/user/siddiquia0/dataset/synpick/"+self.split+"/rgb_new_seq"))


        #for idx in (((self.valid_ids))):
            #self.index.append(idx)
            
        self._find_valid_sequences()
        print("valid", len(self.valid_idx))
        self.create_new_dir(self.valid_idx)
        
        #self.image_valid = [os.path.join("/home/user/siddiquia0/dataset/synpick/"+self.split+"/rgb_new_seq", image_id) for image_id in self.valid_ids]

        #self.masks_valid = [os.path.join("/home/user/siddiquia0/dataset/synpick/"+self.split+"/newmask1_new_seq", image_id) for image_id in self.valid_ids]
        
        #self.masks_valid = [path.replace('.jpg', '.png') for path in self.masks_valid]
        return



    def create_new_dir(self, valid_idx):
        for idx in self.valid_idx:
            
            masks = [imageio.imread(self.mask_fps[idx])]
            if len(masks[0].shape) == 4: 
                masks[0] = masks[0][..., 0]
            masks[0] = torch.from_numpy(masks[0])
            #masks[0] = transforms.Resize(size=self.img_size,interpolation=transforms.InterpolationMode.NEAREST)(masks[0])
            masks[0] = masks[0].unsqueeze(1)
            
            #masks = torch.Tensor(masks)
            #masks = transforms.Resize(self.img_size)(masks)
            head_tail = os.path.split(self.mask_fps[idx])
            transform = T.ToPILImage()
            masks[0]=masks[0].reshape(1,135,240)
            masks = transform(masks[0])
            masks.save("/home/user/siddiquia0/dataset/synpick/"+self.split+"/newmask1_new_seq/"+head_tail[1])
            
            
            imgs1 = [imageio.imread(self.image_fps[idx]) / 255. ]
            # preprocessing and computing segmentation maps
            imgs1 = np.stack(imgs1, axis=0)
            imgs1 = torch.Tensor(imgs1).permute(0, 3, 1, 2)
            imgs1 = transforms.Resize(self.img_size)(imgs1)
            head_tail = os.path.split(self.image_fps[idx])
            transform = T.ToPILImage()
            imgs1=imgs1.reshape(3,135,240)
            img = transform(imgs1)
            img.save("/home/user/siddiquia0/dataset/synpick/"+self.split+"/rgb_new_seq/"+head_tail[1])

            


    def __len__(self):
        """ Number of valid sequences in the dataset """
        return len(self.valid_ids)
                                                                                          
    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        #i = self.index[i]  # only consider valid indices
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        print(seq_len)
        idx = range(i, i + seq_len, self.seq_step)  # create range of indices for frame sequence
        imgs = [imageio.imread(self.image_valid[i])/255.] 
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        print(imgs.shape)
        # preprocessing and computing segmentation maps
        segmentation = self._get_segmentation_maps(
                path=self.masks_valid,
                idx=i
            )
        #instance = self._get_segmentation_maps(path=self.instance_fps,idx=i)

        data = {"frames": imgs, "segmentation": segmentation}
        return data

    def _get_segmentation_maps(self, path, idx):
        """ Loading segmentation maps for sequence idx """
        seg_maps = np.array([imageio.imread(path[idx])])
        if len(seg_maps.shape) == 4: 
            seg_maps = seg_maps[..., 0]
        seg_maps = torch.from_numpy(seg_maps)
        seg_maps = transforms.Resize(size=self.img_size,interpolation=transforms.InterpolationMode.NEAREST)(seg_maps)
        seg_maps = seg_maps.unsqueeze(1)
        return seg_maps

    def _create_frame_to_obj_kpoints_dict(self):
        """
        Loading object poses from scene-ground-truth file into a data structure.
        """
        for scene_gt_fp in self.scene_gt_fps:
            # loading ground truth (meta-)data from an episode
            ep_num = self._ep_num_from_fname(scene_gt_fp)
            with open(scene_gt_fp, 'r') as scene_json_file:
                ep_dict = json.load(scene_json_file)
            # obtaining objects and adding poses
            object_poses = []
            for frame_elem in ep_dict.values():
                frame_obj_dict = defaultdict(list)
                for obj_elem in frame_elem:
                    obj_id = obj_elem["obj_id"]
                    pos_3d = obj_elem["cam_t_m2c"]
                    frame_obj_dict[obj_id].append(pos_3d)
                object_poses.append(frame_obj_dict)
            self.object_poses[ep_num] = object_poses
        return

    def _find_valid_sequences(self):
        """
        Finding valid sequences in the corresponding dataset split.
        The following conditions are enforced:
            - Sequence has the sufficient number of frames
            - No overlap between sequences in validation and test sets
            - Selecting only sequences in which the robot gripper is present
        """
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        print("length", len(self.image_ids))
        last_valid_idx = -1 * seq_len
        frame_offsets = range(0, self.num_frames * self.seq_step, self.seq_step)
        for idx in tqdm(range(len(self.image_ids) - seq_len + 1)):
            # handle overlapping sequences
            """if (not self.allow_seq_overlap and (idx < last_valid_idx + seq_len))                  or (self.allow_seq_overlap and (idx < last_valid_idx + seq_len * self.max_overlap)):
                #print("skipping frame",idx)
                continue"""

            ep_nums = [self._ep_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            frame_nums = [self._frame_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            # first few frames are discarded
            if frame_nums[0] < self.SKIP_FIRST_N:
                #print("skipping part for idx",idx)
                continue
            # last T frames of an episode should not be chosen as the start of a sequence
            if ep_nums[0] != ep_nums[-1]:
                #print("skipping frames", idx)
                continue

            # obtaining the pose of the gripper in the selected frames
            gripper_obj_id = self.NUM_CLASSES
            gripper_obj_id=22
            gripper_pos = [self.object_poses[ep_nums[0]][frame_num][gripper_obj_id][0] for frame_num in frame_nums]

            # discard sequences where the gripper_pos is not low enough (e.g. Gripper still descending)
            offset = self.GRIPPER_VALID_OFFSET
            gripper_xy = self._to_img_plane(gripper_pos[0])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)): 
                ##print("skiiping frames", idx)
                continue
            gripper_xy = self._to_img_plane(gripper_pos[-1])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)):
                #print("skipping frames", idx)
                continue

            # discard sequences without considerable gripper movement
            gripper_pos_deltas = self._get_gripper_pos_xydist(gripper_pos)
            gripper_pos_deltas_above_min = [(delta > 1.0) for delta in gripper_pos_deltas]
            gripper_pos_deltas_below_max = [(delta < 30.0) for delta in gripper_pos_deltas]
            most = lambda lst, factor=0.67: sum(lst) >= factor * len(lst)
            gripper_movement_ok = most(gripper_pos_deltas_above_min) and all(gripper_pos_deltas_below_max)
            if not gripper_movement_ok: 
                #print("skipping frames", idx)
                continue
            self.valid_idx.append(idx)

            last_valid_idx = idx
        print("length is", len(self.valid_idx))

        if len(self.valid_idx) <= 0:
            raise ValueError("No valid sequences were found...")
        return

    def _to_img_plane(self, pos_3d):
        """ Pose3d to pose2d """
        Bin_H, Bin_W = self.BIN_SIZE
        px, py, _ = pos_3d
        x, y = px / Bin_W + 0.5, py / Bin_H + 0.5
        # discard objects dropped outside of the bin
        return (x, y) if ((0 <= x < 1.0) and (0 <= y < 1.0)) else (0., 0.)

    def _comp_gripper_pos(self, old, new):
        x_diff, y_diff = new[0] - old[0], new[1] - old[1]
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)

    def _get_gripper_pos_xydist(self, gripper_pos):
        return [self._comp_gripper_pos(old, new) for old, new in zip(gripper_pos, gripper_pos[1:])]

    def _get_gripper_pos_diff(self, gripper_pos):
        gripper_pos_numpy = np.array(gripper_pos)
        return np.stack([new-old for old, new in zip(gripper_pos_numpy, gripper_pos_numpy[1:])], axis=0)

    def _ep_num_from_id(self, file_id):
        """ Obtaining episode number from RBG image filename """
        return int(file_id[-17:-11])

    def _frame_num_from_id(self, file_id):
        """ Obtaining frame number in episode from RBG image filename """
        return int(file_id[-10:-4])

    def _ep_num_from_fname(self, file_name):
        """ Fetching video/episode name from the scene_gt file name """
        if "scene_gt" not in file_name:
            raise ValueError(f"Given {file_name = } is not a *scene_gt.json file...")
        return int(file_name[-20:-14])

    def _check_num_frames_param(self, num_frames, split):
        """
        Making sure the given 'num_frames' is valid for the corresponding split
        """
        if split == "test" or split == "val":
                num_frames = 1

        return num_frames

SynpickVP("train",10)
#SynpickVP("test",1)
#SynpickVP("val",1)
