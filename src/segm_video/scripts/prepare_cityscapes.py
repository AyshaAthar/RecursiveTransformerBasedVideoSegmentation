"""Prepare Cityscapes dataset"""
import click
import os
import shutil
import mmcv
import zipfile

from pathlib import Path
#from segm_video.utils.download import download

USERNAME = None
PASSWORD = None


def download_cityscapes(path, username, password, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ("gtFine_trainvaltest.zip", "99f532cb1af174f5fcc4c5bc8feea8c66246ddbc"),
        ("leftImg8bit_trainvaltest.zip", "2c0b77ce9933cc635adda307fbba5566f5d9d404"),
    ]
    download_dir = path    
 
    for filename, checksum in _CITY_DOWNLOAD_URLS:
        # extract
        with zipfile.ZipFile(str(download_dir / filename), "r") as zip_ref:
            zip_ref.extractall(path="/home/user/siddiquia0/dataset/")
        print("Extracted", filename)


def install_cityscapes_api():
    os.system("pip install cityscapesscripts")
    try:
        import cityscapesscripts
    except Exception:
        print(
            "Installing Cityscapes API failed, please install it manually %s"
            % (repo_url)
        )


def convert_json_to_label(json_file):
    from cityscapesscripts.preparation.json2labelImg import json2labelImg

    label_file = json_file.replace("_polygons.json", "_labelTrainIds.png")
    json2labelImg(json_file, label_file, "trainIds")


@click.command(help="Initialize Cityscapes dataset.")
@click.argument("download_dir", type=str)
@click.option("--username", default=USERNAME, type=str)
@click.option("--password", default=PASSWORD, type=str)
@click.option("--nproc", default=10, type=int)
def main(
    download_dir,
    username,
    password,
    nproc,
):

    #dataset_dir = Path(download_dir) / "cityscapes_new"


    #download_cityscapes(dataset_dir, username, password, overwrite=False)

    install_cityscapes_api()
    dir="/home/user/siddiquia0/dataset/"
    gt_dir = "/home/user/siddiquia0/dataset/cityscapes_new/gtFine_sequence"

    poly_files = []
    for poly in mmcv.scandir(str(gt_dir), "_polygons.json", recursive=True):
        poly_file = gt_dir +"/"+ poly
        poly_files.append(poly_file)
    mmcv.track_parallel_progress(convert_json_to_label, poly_files, nproc)

    split_names = ["train", "val", "test"]

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(gt_dir +"/"+ split, "_polygons.json", recursive=True):
            filenames.append(poly.replace("_gtFine_polygons.json", ""))
        with open(dir +  "/"+ f"{split}.txt", "w") as f:
            f.writelines(f + "\n" for f in filenames)


if __name__ == "__main__":
    main()