from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from pathlib import Path
from segm_video.data.base import BaseMMSeg
from segm_video.data import utils
from segm_video.config import dataset_dir
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



SYNPICK_CONFIG_PATH = "/home/user/siddiquia0/segmenter/segm/data/config/synpick.py"

SYNPICK_CATS_PATH = "/home/user/siddiquia0/segmenter/segm/data/config/synpick.yml"

@DATASETS.register_module()
class synpick(CustomDataset):
    classes=("background","master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
                  "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
                  "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
                  "large_clamp", "extra_large_clamp", "foam_brick", "gripper")

    palette=[[0,0,255],[0,128,0],[128, 128, 0],[255, 0, 0],[255, 255, 0],[128,0,128],[125,165,0],
         [244,164,96],[165,42,42],[255,192,203],[139,64,0],[218,165,32],[34,139,34],[0,255,127],
         [0,255,255],[65,105,255],[0,0,128],[148,0,211],[103,49,71],[255,0,255],[112,128,144],[128,0,0],[0,0,0]]
    CLASSES = classes
    PALETTE = palette
    
    
    def __init__(self, split, **kwargs):
        """if(split=='train'):
            img_dir = '/home/user/siddiquia0/dataset/synpick/train/rgb'
            ann_dir = '/home/user/siddiquia0/dataset/synpick/train/newmask1'
        else:
            img_dir= '/home/user/siddiquia0/dataset/synpick/test/rgb'
            ann_dir = '/home/user/siddiquia0/dataset/synpick/test/newmask1'"""
        split=None
        

        super().__init__(split=split,img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        
        def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
            """Load annotation from directory.

            Args:
                img_dir (str): Path to image directory
                img_suffix (str): Suffix of images.
                ann_dir (str|None): Path to annotation directory.
                seg_map_suffix (str|None): Suffix of segmentation maps.
                split (str|None): Split txt file. If split is specified, only file
                    with suffix in the splits will be loaded. Otherwise, all images
                    in img_dir/ann_dir will be loaded. Default: None

            Returns:
                list[dict]: All image info of dataset.
            """

            img_infos = []
            if split is not None:
                lines = mmcv.list_from_file(
                    split, file_client_args=self.file_client_args)
                for line in lines:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
            else:
                for img in self.file_client.list_dir_or_file(
                        dir_path=img_dir,
                        list_dir=False,
                        suffix=img_suffix,
                        recursive=True):
                    img_info = dict(filename=img)
                    if ann_dir is not None:
                        seg_map = img.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
                img_infos = sorted(img_infos, key=lambda x: x['filename'])
            
            print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
            return img_infos
        
        
class SynpickVP(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        
        super().__init__(image_size, crop_size, split, SYNPICK_CONFIG_PATH,**kwargs)
        self.names, self.colors = utils.dataset_cat_description(SYNPICK_CATS_PATH)
        self.n_cls = 23
        self.reduce_zero_label = True

    def update_default_config(self, config):

        root_dir = dataset_dir()
        path = Path(root_dir) / "synpick"
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
