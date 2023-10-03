import segm_video.utils.torch as ptu

from segm_video.data import ImagenetDataset
#from segm_video.data import ADE20KSegmentation
#from segm_video.data import PascalContextDataset
from segm_video.data import CityscapesDataset
"""from segm_video.data import SynpickVP
from segm_video.data import synpick"""
from segm_video.data import Loader
from segm_video.data import SynpickseqVP
from segm_video.data import CityscapesseqVP
def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    # load dataset_name
    if dataset_name == "imagenet":
        dataset_kwargs.pop("patch_size")
        dataset = ImagenetDataset(split=split, **dataset_kwargs)
    elif dataset_name == "ade20k":
        dataset = ADE20KSegmentation(split=split, **dataset_kwargs)
    elif dataset_name == "pascal_context":
        dataset = PascalContextDataset(split=split, **dataset_kwargs)
    elif dataset_name == "cityscapes":
        dataset = CityscapesDataset(split=split, **dataset_kwargs)
    elif dataset_name == "synpickseq1":
        if(split=="train"):
            dataset=SynpickseqVP(split="train",**dataset_kwargs)
        elif(split=="test"):
            dataset=SynpickseqVP(split="test",**dataset_kwargs)
        else:
            dataset=SynpickseqVP(split="val",**dataset_kwargs)
    elif dataset_name == "cityscapesseq":
        if(split=="train"):
            dataset=CityscapesseqVP(split="train",**dataset_kwargs)
        elif(split=="test"):
            dataset=CityscapesseqVP(split="test",test_mode=True,**dataset_kwargs)
        else:
            dataset=CityscapesseqVP(split="val",test_mode=True,**dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
