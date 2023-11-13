

from torchvision.transforms.functional import convert_image_dtype
# import cv2
import numpy as np
import os
import tqdm
import pickle
import json
import sys
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from src.video_reader import custom_read_video
import torchvision.io



def inference(inputs,model,transformation, config):

    for video in inputs:
        stream = "video"
        video_reader = torchvision.io.VideoReader(video, stream)
        video_frames, _, pts, meta = custom_read_video(video_reader)
        frame_n = meta["num_frames"]
        prog_bar = tqdm.tqdm(total=frame_n)
        outputs_all = {}
        for frame_count, frame in enumerate(video_frames, start= 1):
            outputs_all[video][frame_count] = model([transformation(frame)])
            prog_bar.update(1)
    return outputs_all