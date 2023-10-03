# dataset settings
dataset_type = "synpick"
data_root = "/home/user/siddiquia0/dataset/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
img_scale = (240, 136)
crop_size = (224, 224)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        flip=False,
        transforms=[

            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
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
        img_dir=["train/rgb","val/rgb_new"],
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
