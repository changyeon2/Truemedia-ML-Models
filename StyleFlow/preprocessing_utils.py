# 1. crop and align face image #
import os
from os.path import join
import cv2
import numpy as np
import torch
import logging
import datetime
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.utils import get_crop_box
from test_tools.dataset_clips import TestClips

from PIL import Image


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device="cuda" if torch.cuda.is_available() else "cpu"

now = datetime.datetime.now()
logger = logging.getLogger("main") 
stream_handler = logging.StreamHandler() 
formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


import re
def remove_extension(filename):
    return re.sub(r'\.(mp4|mov|webm|avi|mpg|mpeg)$', '', filename)

def crop_face_from_video(video_path, clip_size=32, max_frames=288, data_root = "/data"):
    """_summary_

    Args:
        video_path (_str_): path of video
        cache_root (_str_): where to save cache file
        crop_root  (_str_): where to save crop image
        clip_size  (_int_): align clip size
    """   
    crop_align_func = FasterCropAlignXRay(256)
    # only .mp4 
    if 'mp4' not in video_path : return
    
    # video name
    video_name = remove_extension(video_path.split('/')[-1])
    os.makedirs(join(data_root,video_name,"crop_images"), exist_ok=True)
    
    if len(os.listdir(join(data_root,video_name,"crop_images")))>=max_frames:
        logger.info(f'{video_name} already croped!')
        return
    
    ### cache_file code delete ###
    
    detect_res, all_lm68, frames = detect_all(
                video_path, return_frames=True, max_size=max_frames
            )
    
    # if there is no frame in the video -> save error list 
    try:
        shape = frames[0].shape[:2]
    except IndexError:
        raise Exception("there is non frames in the video!")
    
    # all detect_res
    all_detect_res = []

    assert len(all_lm68) == len(detect_res)
    
    # in each frame, save the detected face's bounding box, landmark(5, 68), score as a tuple and save it in a list
    for faces, faces_lm68 in zip(detect_res, all_lm68):
            new_faces = []
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_face = (box, lm5, face_lm68, score)
                new_faces.append(new_face)
            all_detect_res.append(new_faces)
    detect_res = all_detect_res    
    
    # SORT tracking
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    
    # if there is no face detected, find the longest face in the video
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)
        
    data_storage = {}
    frame_boxes = {}
    super_clips = []
    frame_res = {}
    super_clips_start_end = []
   
    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)): 
        
        # if detect_res's length is not equal to track's length, raise error 
        assert len(detect_res[start:end]) == len(track)
        
        super_clips.append(len(track))
        super_clips_start_end.append((start, end))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))): 
            box,lm5,lm68 = face[:3] # box, lm5, lm68
            big_box = get_crop_box(shape, box, scale=0.5) # get crop box

            top_left = big_box[:2][None, :] # top left point

            new_lm5 = lm5 - top_left 
            new_lm68 = lm68 - top_left

            new_box = (box.reshape(2, 2) - top_left).reshape(-1)

            info = (new_box, new_lm5, new_lm68, big_box) # face info


            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            # cropped = cv2.resize(cropped, (512, 512))

            base_key = f"{track_i}_{j}_" # i : face, j : frame
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(np.int)
    
    logger.info(f"{video_name}  :  sampling clips from super clips {super_clips}")
    clips_for_video = []
    clip_size = clip_size
    pad_length = clip_size - 1
    
    # if video include multi-face
    # if len(super_clips) > 1:
    #     f = open(join(cache_root, f"manyface_{video_name}.txt"), 'a')
    #     line = f"{video_name} : "
    #     for start, end in super_clips_start_end:
    #         line += f"{start}-{end}, "
    #     line += '\n'
    #     f.write(line)
    #     f.close()
        
    # make clip by each face id
    # cut the super clip into clips, overlap 7frames, 8frames per clip 
    for super_clip_idx, super_clip_size in enumerate(super_clips): 
        inner_index = list(range(super_clip_size))
        
        # if there is not enough frames to make a clip, pad the frames
        if super_clip_size < clip_size:    
    
            if super_clip_size < clip_size//2 : continue 
            post_module = inner_index[1:-1][::-1] + inner_index

            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            assert len(post_module) == pad_length

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            assert len(pre_module) == pad_length

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)

        frame_range = [
            inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)
            
    # all_frames = {}
    # landmarks, images = crop_align_func(landmarks, images) # i : face, j : frame
    for clip in clips_for_video: 
        # call cropped face images from data_storage, i : face, j : frame
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip] 
        # call landmarks from data_storage, i : face, j : frame
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip] 

        landmarks, images = crop_align_func(landmarks, images) # align the face images by landmarks in the clip
        i, j = clip[-1]
        k = super_clips[i]%clip_size 
        
        # if last frame number of the clip is multiple of clip_size, save all images in the clip
        if (j+1)%clip_size==0: 
            # it means save face alignments in 8 frames in the video so that they don't overlap
            for f, (i,j) in enumerate(clip) :
                # only save first person face images
                if i == 0:
                    # print(f'{i:02}_{j:04}.png')
                    cv2.imwrite(join(data_root,video_name, f'crop_images/{i:02}_{j:04}.png'), cv2.cvtColor(images[f], cv2.COLOR_BGR2RGB))
                    # FIXME #
                    # all_frames[f'{i:02}_{j:04}'] = cv2.cvtColor(images[f], cv2.COLOR_BGR2RGB)
                else:
                    pass
        # if the clip have last frame image, save all images in the clip
        if j == super_clips[i]-1: 
            # if the clip is not multiple of clip_size, save the last k images in the clip
            if k!=0 : 
                # k is the number of frames that are not overlapped
                for l in range(clip_size-k,clip_size):
                    ci,cj = clip[l]
                    # only save first person face images
                    if ci == 0:
                        # print(f'{ci:02}_{cj:04}.png')
                        # clip means face alignment images in 8 frames, non-overlap
                        cv2.imwrite(join(data_root,video_name, f'crop_images/{ci:02}_{cj:04}.png'), cv2.cvtColor(images[l], cv2.COLOR_BGR2RGB))
                        # FIXME  #
                    else:
                        pass
    # memory flush #
    clips_for_video = None
    crop_align_func = None
    return 

# set default json

def set_result():
    return {
        "video": {
            "name": [],
            "df_probability": [],
            "prediction": [],
        }
    }


import argparse
if __name__ == '__main__':
    
    pass
        
        

    
