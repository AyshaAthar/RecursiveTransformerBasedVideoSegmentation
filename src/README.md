# How to run code?
## Code structure -> 
```
Dataset directory/  
Working directory/
        ├── segm_video
           ├── model
           ├── data
```

Data structure
My dataset directory - /home/user/siddiquia0/dataset/
```
Dataset directory/
        ├── synpickseq
           ├── train
               ├── rgb
               ├── segmentation
           ├── test
               ├── rgb
               ├── segmentation
       ├──cityscapes
          ├──leftImg8bit_sequence
              ├──train
                 ├──aachen
              ├──test
                 ├──bonn
          ├──gtFine_sequence
              ├──train
                 ├──aachen
              ├──test
                 ├──bonn
```
Set environment variable dataset pointing to the directory containing the dataset by the command -
 ```sh
export DATASET=/home/user/siddiquia0/dataset
```
Next - in the data config file -> segm_video/data/config/synpickseq.py 

Set appropriate paths to data_root, img_dir, ann_dir for the pipeline.

Set dataset_type to the name of the datasets parent folder, for eg: for synpick, set synpickseq

In order to set batch_size, learning rate, etc, set them in the main config file -> segm_video/config.yml

## To run training scripts
Start at the src folder

Run command - 

```sh
gpu run python -m train --log-dir synpickseq_linear_mse --dataset synpickseq1 --backbone vit_tiny_patch16_384 --decoder linear --pretrain-log-dir synpick_tiny_linear
```
where
1. log-dir : Directory to log the results, store checkpoints.
2. dataset : Name of dataset, use synpickseq for synpick and cityscsapesseq for cityscapes.
3. backbone: ViT encoder backbone.
4. decoder: Type of decoder, linear/mask_transformer.
5. pretrain-log-dir: Directory from which it loads pretrained frame-by-frame model.

## For Evaluation 
Run command :
```sh
gpu run python -m evaluationscript --model-path synpickseq_linear_mse --dataset-name synpickseq1
```
where
1. model-path: Directory that contains the checkpoints. Have the directory inside the main folder.
2. dataset-name : Name of dataset, use synpickseq for synpick and cityscsapesseq for cityscapes.

