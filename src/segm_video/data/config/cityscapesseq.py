# dataset settings
from segm_video.data import LoadImageFromFile1,LoadAnnotations1,RandomCrop1,RandomFlip1,PhotoMetricDistortion1,Normalize1,Pad1,DefaultFormatBundle1,Collect1,MultiScaleFlipAug1,ImageToTensor1
dataset_type = "cityscapes_new"
data_root = "/home/user/siddiquia0/dataset/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (768, 768)
max_ratio = 2
"""train_pipeline = [
    dict(type="LoadImageFromFile1"),
]"""
train_pipeline = [
    dict(type="LoadImageFromFile1"),
    dict(type="LoadAnnotations1"),
    dict(type="Resize1", img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop1", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip1", prob=0.5),
    dict(type="PhotoMetricDistortion1"),
    dict(type="Normalize1", **img_norm_cfg),
    dict(type="Pad1", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle1"),
    dict(type="Collect1", keys=['img', 'sequence_imgs', 'gt_semantic_seg', 'colors']),
]
val_pipeline = [
    dict(type="LoadImageFromFile1"),
    dict(
        type="MultiScaleFlipAug1",
        img_scale=(1024 * 2, 1024),
        flip=False,
        transforms=[
            dict(type="Resize1", keep_ratio=True),
            dict(type="RandomFlip1"),
            dict(type="Normalize1", **img_norm_cfg),
            dict(type="ImageToTensor1", keys=["sequence_imgs"]),
            dict(type="Collect1", keys=["sequence_imgs","ann_dir"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile1"),
    dict(
        type="MultiScaleFlipAug1",
        img_scale=(1024 * max_ratio, 1024),
        flip=False,
        transforms=[
            dict(type="Resize1", keep_ratio=True),
            dict(type="RandomFlip1"),
            dict(type="Normalize1", **img_norm_cfg),
            dict(type="ImageToTensor1", keys=["sequence_imgs"]),
            dict(type="Collect1", keys=["sequence_imgs"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="leftImg8bit/train",
        ann_dir="gtFine_sequence/train",
        sequence_dir = "leftImg8bit_sequence/train",
        split = "train",
        pipeline=train_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["leftImg8bit/train", "leftImg8bit/val"],
        sequence_dir = ["leftImg8bit_sequence/train", "leftImg8bit_sequence/val"],
        ann_dir=["gtFine_sequence/train", "gtFine_sequence/val"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="leftImg8bit/val",
        ann_dir="gtFine_sequence/val",
        sequence_dir = "leftImg8bit_sequence/val",
        split="val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="leftImg8bit/test",
        ann_dir="gtFine_sequence/test",
        sequence_dir = "leftImg8bit_sequence/test",
        split="test",
        pipeline=test_pipeline,
    ),
)
