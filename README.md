# Segmentation-On-a-Video-
## Description
A script to preform a segmentation on a given video - a mask per frame of the video.
This code uses [VisTR repo](https://github.com/Epiphqny/VisTR): End-to-End Video Instance Segmentation with Transformers, for getting the masks of the video.

## Installation
For this script to work properly, you need an Nvidia GPU. 

Next download and compile the code from [VisTR repo](https://github.com/Epiphqny/VisTR) by following the Istructions on thier repo.

Then, download this repo's script and place it on the 'vistr' folder.

If you are having an error about GPU out of memory try adding the following line at the first line of the for loop that iterates the vis_num indices on the interface.py file of [VisTR repo](https://github.com/Epiphqny/VisTR).
```
torch.cuda.empty_cache()
```
## Usage
### Interface
Here is an exsample of the script usage with some arguments
```
python get_masks_of_video.py --masks --model_path /home/soul/PreWorkProject/vistr/vistr_r101.pth --img_path /home/soul/PreWorkProject/rawFrames --ann_path /home/soul/PreWorkProject/vistr/data/annotations/input.json --mp4_clip_path 20201025_172452_II_clip.mp4
```
### Useful Arguments
* --masks
* --model_path - path for the pretrained model (you can download the model )
