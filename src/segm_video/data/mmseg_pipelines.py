import mmcv
from mmcv.utils import deprecated_api_warning
from mmseg.datasets import PIPELINES
from mmcv.parallel import DataContainer as DC 
import collections
from collections.abc import Sequence 
from mmseg.datasets.builder import DATASETS
import numpy as np
from mmcv.utils import build_from_cfg
import os.path as osp
import torch
import sys

@PIPELINES.register_module()
class LoadImageFromFile1(object):
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2' """ 
    
    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """ 
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        if results.get('sequence_prefix') is not None:
            sequence_imgs = []
            sequence = results['sequence_info']['frames']
            for f in sequence:
                frame_filename = osp.join(results['sequence_prefix'],
                                          f)
                img_bytes = self.file_client.get(frame_filename)
                frame = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    frame = frame.astype(np.float32)
                sequence_imgs.append(frame)
        else:
            sequence = None
            sequence_imgs = None
        results['ann_dir'] = results['seg_prefix']
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['sequence_filename'] = sequence
        results['sequence_imgs'] = sequence_imgs
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotations1(object):
    """Load annotations for semantic segmentation.
    Args:
        reduct_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:

            """ filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)""" 
        if results.get('sequence_prefix') is not None:
            sequence_gt_semantic_seg = []
            sequence = results['ann_info']['seg_map']
            if(isinstance(sequence,str)):
                frame_filename = osp.join(results['seg_prefix'],
                                          sequence)
                img_bytes = self.file_client.get(frame_filename)
                frame = mmcv.imfrombytes(
                    img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)
                # modify if custom classes
                if results.get('label_map', None) is not None:
                    for old_id, new_id in results['label_map'].items():
                        frame[frame == old_id] = new_id
                # reduce zero_label
                if self.reduce_zero_label:
                    # avoid using underflow conversion
                    frame[frame == 0] = 255
                    frame = frame - 1
                    frame[frame == 254] = 255    

                sequence_gt_semantic_seg.append(frame)
            else:    
                for f in sequence:
                    frame_filename = osp.join(results['seg_prefix'],
                                              f)
                    img_bytes = self.file_client.get(frame_filename)
                    frame = mmcv.imfrombytes(
                        img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)
                    # modify if custom classes
                    if results.get('label_map', None) is not None:
                        for old_id, new_id in results['label_map'].items():
                            frame[frame == old_id] = new_id
                    # reduce zero_label
                    if self.reduce_zero_label:
                        # avoid using underflow conversion
                        frame[frame == 0] = 255
                        frame = frame - 1
                        frame[frame == 254] = 255    

                    sequence_gt_semantic_seg.append(frame)
        else:
            sequence = None
            sequence_gt_semantic_seg = None

        # modify if custom classes
        """if results.get('label_map', None) is not None:
            print("here")
            sys.exit(1)
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        else:
            print("hi")
            sys.exit(1)
        # reduce zero_label

        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255""" 
        results['gt_semantic_seg'] = sequence_gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class Collect1(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor',  'img_norm_cfg', 'sequence_filename','ann_dir')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """ 
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = (img_meta)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ +                f'(keys={self.keys}, meta_keys={self.meta_keys})'
 
@PIPELINES.register_module()
class ImageToTensor1(object):
    """Convert image to :obj:`torch.Tensor` by given keys.
    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).
    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.
        Args:
            results (dict): Result dict contains the image data to convert.
        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            if key == 'sequence_imgs':
                sequence_imgs = []
                for img in results[key]:
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                    sequence_imgs.append(img)
                results['sequence_imgs'] = to_tensor(sequence_imgs)
            else:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results[key] = to_tensor(img.transpose(2, 0, 1))     
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
 
@PIPELINES.register_module()
class MultiScaleFlipAug1(object):
    """Test-time augmentation with multiple scales and flipping.
    An example configuration is as followed:
    .. code-block::
        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose1(transforms)
        if img_ratios is not None:
            # mode 1: given a scale and a range of image ratio
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
            assert isinstance(img_scale, tuple) and len(img_scale) == 2
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 2: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple)
        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
    
@PIPELINES.register_module()
class RandomFlip1(object):
    """Flip the image & seg.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction']) 

            # flip sequence
            results['sequence_imgs'] = [mmcv.imflip(
                img, direction=results['flip_direction']) for img in results['sequence_imgs']]

            # flip segs
            #for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                #results[key] = mmcv.imflip(
                    #results[key], direction=results['flip_direction']).copy()
                
                # flip sequence
            results['gt_semantic_seg'] = [mmcv.imflip(
                img, direction=results['flip_direction']) for img in results['gt_semantic_seg']]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class Pad1(object):
    """ Pad the image & mask.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """ 

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """ Pad images according to ``self.size``.""" 
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_sequence(self, results):
        #Pad images according to ``self.size``.
        if self.size is not None:
            padded_sequence_imgs = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['sequence_imgs']]
        elif self.size_divisor is not None:
            padded_sequence_imgs = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['sequence_imgs']]

        results['sequence_imgs'] = padded_sequence_imgs

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``.""" 
        for key in results.get('seg_fields', []):
            results[key] = [mmcv.impad(
                img,
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val) for img in results[key]]

    def __call__(self, results):
        """ Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """ 
        self._pad_img(results)
        self._pad_sequence(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, '                     f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class Normalize1(object):
    """ Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """ 

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['sequence_imgs'] = [mmcv.imnormalize(img, self.mean, self.std,
                                                     self.to_rgb) for img in results['sequence_imgs']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb='                     f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class RandomCrop1(object):
    """Random crop the image & seg.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = [self.crop(img, crop_bbox) for img in results['gt_semantic_seg']]
                
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop the sequence
        results['sequence_imgs'] = [self.crop(img, crop_bbox) for img in results['sequence_imgs']]

        # crop semantic seg
        #for key in results.get('seg_fields', []):
            #results[key] = [self.crop(img, crop_bbox) for img in results[key]]   
        results['gt_semantic_seg'] = [self.crop(img, crop_bbox) for img in results['gt_semantic_seg']]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


from numpy import random
@PIPELINES.register_module()
class PhotoMetricDistortion1(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def distortion(self, img):
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        # img distortion
        results['img'] = self.distortion(results['img']) 

        # sequence_imgs distortion
        results['sequence_imgs'] = [self.distortion(img) for img in results['sequence_imgs']]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        data = np.array(data)
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')  
 
@PIPELINES.register_module()
class Resize1(object):
    """ Resize images & seg.
    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.
    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.
    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.
    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """ 

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """ Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """ Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """ 

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """ Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """ 

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """ 

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        #Resize images with ``results['scale']``.
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_sequence(self, results):
        #Resize images with ``results['scale']``.
        if self.keep_ratio:
            #results['sequence_imgs'] = [mmcv.imresize(
                #img, results['scale'],interpolation='nearest', return_scale=False) for img in results['sequence_imgs']]
            
            results['sequence_imgs'] = [mmcv.imrescale(
                img, results['scale'], return_scale=False) for img in results['sequence_imgs']]
        else:
            #results['sequence_imgs'] = [mmcv.imresize(
                #img, results['scale'],interpolation='nearest', return_scale=False) for img in results['sequence_imgs']]
            results['sequence_imgs'] = [mmcv.imresize(
                img, results['scale'], return_scale=True) for img in results['sequence_imgs']]

    def _resize_seg(self, results):
        #Resize semantic segmentation map with ``results['scale']``.
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                #results[key] = [mmcv.imresize(
                #img, results['scale'], interpolation='nearest') for img in results['gt_semantic_seg']]

                #results[key], scale_factor = [mmcv.imrescale(img, results['scale'], interpolation='nearest') for img in results['gt_semantic_seg']]
                results[key] = [mmcv.imrescale(img, results['scale'], interpolation='nearest') for img in results['gt_semantic_seg']]
            else:
                results[key] = [mmcv.imresize(
                img, results['scale'], interpolation='nearest') for img in results['gt_semantic_seg']]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            self._random_scale(results)
        if 'img' in results:
            self._resize_img(results)
        self._resize_sequence(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str
    
@PIPELINES.register_module()
class DefaultFormatBundle1(object):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            sequence_gts = []
            for img in results['gt_semantic_seg']:
                img = img[None,...].astype(np.int64)
                sequence_gts.append(img)
            results['gt_semantic_seg'] = DC(to_tensor(sequence_gts), stack=True)

        if 'sequence_imgs' in results:
            sequence_imgs = []
            for img in results['sequence_imgs']:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                sequence_imgs.append(img)
            results['sequence_imgs'] = DC(to_tensor(sequence_imgs), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Compose1(object):
    """Compose multiple transforms sequentially.
    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string    