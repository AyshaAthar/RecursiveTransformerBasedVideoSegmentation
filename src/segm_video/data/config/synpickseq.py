# dataset settings
from segm_video.data import LoadImageFromFile1,LoadAnnotations1,RandomCrop1,RandomFlip1,PhotoMetricDistortion1,Normalize1,Pad1,DefaultFormatBundle1,Collect1,MultiScaleFlipAug1,ImageToTensor1
dataset_type = "synpickseq1"
data_root = "/home/user/siddiquia0/dataset"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

img_scale = (240, 136)
crop_size = (128, 128)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromFile1"),
    dict(type="LoadAnnotations1", reduce_zero_label=True),
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
        flip=False,
        transforms=[
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
        flip=False,
        transforms=[
            dict(type="RandomFlip1"),
            dict(type="Normalize1", **img_norm_cfg),
            dict(type="ImageToTensor1", keys=["sequence_imgs"]),
            dict(type="Collect1", keys=["sequence_imgs"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/rgb",
        ann_dir = "train/newmask1_new",
        pipeline=train_pipeline,
    split="train",

    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["train/rgb","val/rgb"],
        ann_dir = ["train/newmask1_new","val/newmask1_new"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/rgb_new",
        ann_dir = "val/newmask1_new",
    split="val",
        pipeline=val_pipeline,

    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="test/rgb",
        ann_dir = "test/newmask1_new",
        pipeline=test_pipeline,
    split="test"
    ),

)
