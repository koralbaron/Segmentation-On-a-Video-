import cv2
import json
import os
import shutil
from pycocotools import mask
import inference
import numpy as np

WHITE = (1,1,1)
COLORS = [(1,1,0),
          (1,0,1),
          (1,0,0),
          (0,1,0),
          (0,0,1),
          (0,1,1),
          (1,1,1),
          (0.5,1,1),
          (0.3,0.5,1),
          (1,0.5,1),
          (0.1,0.4,0.5),
          (0,0.9,0.7),
          (0.3,0,0.9),
          (0.3,0.5,0.9),
          (0.4,0,0),
          (0.4,0,0.7),
          (0.4,0.2,0.1),
          (0.4,0.4,0.3),
          (0.5,0,0.3),
          (0.4,1,0.7),
          (0.4,0.9,0.8),
          (0.6,0.5,0.5),
          (0.6,0.9,1),
          (0.7,0,0.3),
          (0.7,0.5,1),
          (0.7,0.6,0.6),
          (0.8,0.4,0.2),
          (0.8,1,0.1),
          (0.8,1,0.8),
          (0.8,0.8,0.9),
          (0.9,0.4,0.8),
          (0.9,1,0.9),
          (0.9,0.8,1),
          (0.9,0.9,0.6),
          (1,1,0.9),
          (1,0.9,0.7),
          (0.4,0.4,0),
          (0.4,0.4,0.6),
          (0.3,0.7,0.1),
          (0.3,0.2,0.7),
          (0.3,0.2,0),
]
def cleanFolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# splits the video into frames
def splitVideoToFrames(vidCap, imgPath):
    print("Spliting Video To Frames...")
    success, image = vidCap.read()
    count = 0
    while success:
        cv2.imwrite(imgPath + "/%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidCap.read()
        count += 1

def createVidDict(id, width, height): 
    vid = {
        "width": width,
        "length": 0,
        "date_captured": "",
        "license": 1,
        "flickr_url": "",
        "file_names": [],
        "id": id,
        "coco_url": "",
        "height": height
    }
    return vid

# creates Json for the video's frames
def createJson(vidCap, imgPath, inputJsonPath):
    print("Creating Json...")
    fileNamesList = os.listdir(imgPath)
    width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = {
        'videos': [
            {
                "width": int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "length": 0,
                "date_captured": "",
                "license": 1,
                "flickr_url": "",
                "file_names": [],
                "id": 1,
                "coco_url": "",
                "height": int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }]}
    framesCounter = 0
    index = 0
    for filename in sorted(fileNamesList, key=lambda x: int(x.split(".")[0])):
        if framesCounter <= 35:
            data['videos'][index]['file_names'].append(filename)
            framesCounter +=1
        else:# new index of every 36 frames
            data['videos'][index]['length'] = 36
            framesCounter = 0
            index +=1
            vidDict = createVidDict(index+1, width, height)
            data['videos'].append(vidDict)
            data['videos'][index]['file_names'].append(filename)
    data['videos'][-1]['length'] = len(data['videos'][-1]['file_names'])

    with open(inputJsonPath, 'w', encoding='utf-8') as jsonFile:
        json.dump(data, jsonFile, indent=4)

def getModelResult(args):
    print("Getting Model Results..")
    inference.main(args)

def setColor(colorValues, b, g , r):
    blue, green, red = colorValues
    np.multiply(b, blue, out=b, casting="unsafe")
    np.multiply(g, green, out=g, casting="unsafe")
    np.multiply(r, red, out=r, casting="unsafe")
    return b,g,r

def catAndSaveImgMask(fileName, masksList2, masksResultsPath, masksColor):
    for i in range(36): # max frame 
        frame = masksList2[0][i]
        for j in range(1,len(masksList2)):
            if frame is None:
                if masksList2[j][i] is None:
                    continue
                else:
                    frame = masksList2[j][i]
            elif masksList2[j][i] is not None:
                mFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mFrame = cv2.bitwise_not(mFrame)
                frame = cv2.add(masksList2[j][i], frame, mask=mFrame)
        if frame is not None:
            if masksColor == False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(masksResultsPath + "/%d.jpg" %fileName, frame) 
            fileName +=1
    return fileName

# decodes the results and saves the img masks in a folder 
def decodeResult(masksResultsPath, threshold, masksColor):
    print("Decoding Result...")
    with open("results.json","r") as jsonFile:
        data = json.load(jsonFile)
    fileName = 0 # fileName - for setting image file name to be index
    masksList = []
    masksList2 = [] # list of masksList
    prevVidID = data[0]['video_id']
    for vidIndex in range(len(data)):
        categoryId = data[vidIndex]['category_id']
        videoId = data[vidIndex]['video_id']
        if videoId != prevVidID:
            fileName = catAndSaveImgMask(fileName, masksList2, masksResultsPath, masksColor)
            masksList2.clear()
        if data[vidIndex]['score'] > threshold:
            for res in data[vidIndex]['segmentations']:
                if not res == None:
                    myMask = mask.decode(res)
                    imgMask = myMask * 255 # for graysacle
                    imgMask = cv2.cvtColor(imgMask,cv2.COLOR_GRAY2RGB)
                    b, g, r = cv2.split(imgMask)
                    b,g,r = setColor(COLORS[categoryId - 1], b, g, r)
                    imgMaskColored  = cv2.merge([b, g, r])
                    masksList.append(imgMaskColored)
                else:
                    masksList.append(None)
        for i in range(36 -len(masksList)):# complete all masksList to 36 items
            masksList.append(None)
        
        masksList2.append(masksList[:])
        masksList.clear() 
        prevVidID = videoId

    catAndSaveImgMask(fileName, masksList2, masksResultsPath, masksColor)

#get masks from given video. Input: video clip, Output: all the img masks of the given video
def main(args):
    if args.clean:
        cleanFolder(args.img_path)
        cleanFolder(args.masks_results_path)
    vidCap = cv2.VideoCapture(args.mp4_clip_path)
    splitVideoToFrames(vidCap, args.img_path)
    createJson(vidCap, args.img_path, args.input_json_path)
    getModelResult(args)
    decodeResult(args.masks_results_path, args.threshold, args.masks_color)

if __name__== "__main__":
    parser = inference.get_args_parser()# Atention: more args can be found ion inference.py
    parser.add_argument("--mp4_clip_path", default="20201025_172452_II_clip.mp4")
    parser.add_argument("--input_json_path", default="data/annotations/input.json")
    parser.add_argument("--masks_results_path", default="data/result_masks")
    parser.add_argument("--threshold", default=0.8, type=float, help="threshold for object prediction")
    parser.add_argument("--masks_color", default= False, type=bool, help="True - different color for deferent class. False - masks are only white")
    parser.add_argument("--clean", default=False, type=bool, help="remove old data from masks_results_path, img_path")
    args = parser.parse_args()
    main(args)
