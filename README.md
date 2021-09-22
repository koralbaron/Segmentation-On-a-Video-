# Segmentation On a Video
## Description
A script to preform a segmentation on a given video - a mask per frame of the video.
This code uses [VisTR repo](https://github.com/Epiphqny/VisTR): End-to-End Video Instance Segmentation with Transformers, for getting the masks of the video.
The code can handle videos that contains up to 10 instances and represent semantic segmentation (a semantic colored mask per frame of the video) or regular black and white mask per frame of the video.

**Input:** MP4 video clip.
**Output:** Segmentation results - mask per frame of the video.
 
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
* ```--masks``` - Train segmentation head if the flag is provided
* ```--model_path``` - Path for the pretrained model (you can download the model from [VisTR repo](https://github.com/Epiphqny/VisTR)
* ```--img_path``` - Path where to save the raw frames of the video (the script will save the images there)
* ```ann_path``` - Path where to save the input.json that the script builds.
* ```mp4_clip_path``` - path for the mp4 video clip input for the script.
* ```--save_path``` - Path path where to save the result.json file.
* ```--backbone``` - Name of the convolutional backbone to use.
* ```--threshold``` - Threshold for object prediction.
* ```masks_color``` - True - different color for deferent class. False - masks are only white.
* ```--clean``` - Remove old data from masks_results_path.

**For more additional argumets refer to inference.py on [VisTR repo](https://github.com/Epiphqny/VisTR)**

## Examples
### Horse Riding - Segmentation clip VS Original clip
A video clip that made from the segmentation results of the script with a horse riding video as input.

Link for the Original video clip: [Horse Galloping in Slow Motion](https://www.youtube.com/watch?v=MvhRgJ9-7Rk), all credits for the original video goes to their channel.

<img src="HorseRiding(1).gif" width="30%" height="400" />
