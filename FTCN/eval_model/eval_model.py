from eval_model._base import ModelBase
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

class EvalModel(ModelBase):
    def __init__(self, device, config="ftcn_tt.yaml", weights='checkpoints/ftcn_tt.pth', max_frames=400, batch_size=8, face_thres=0.75, bbox_thres=5000):
        super().__init__(device)
        # Add additional attributes for inference here
        self.cfg = cfg
        self.cfg.init_with_yaml()
        self.cfg.update_with_yaml(config)
        self.cfg.freeze()

        self.crop_align_func = FasterCropAlignXRay(self.cfg.imsize)
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.face_thres = face_thres
        self.bbox_thres = bbox_thres

        self.model = PluginLoader.get_classifier(self.cfg.classifier_type)()
        self.model.load(weights)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_path):
        # Implement inference logic here
        cache_file = f"{input_path}_{self.max_frames}.pth"

        if os.path.exists(cache_file):
            detect_res, all_lm68 = torch.load(cache_file)
            frames = grab_all_frames(input_path, max_size=self.max_frames, cvt=True)
            # print("detection result loaded from cache")
        else:
            # print("detecting")
            detect_res, all_lm68, frames = detect_all(input_path, return_frames=True, max_size=self.max_frames, face_thres=self.face_thres, bbox_thres=self.bbox_thres)
            torch.save((detect_res, all_lm68), cache_file)
            # print("detect finished")

        # print("number of frames: ", len(frames))

        shape = frames[0].shape[:2]

        all_detect_res = []

        assert len(all_lm68) == len(detect_res)

        for faces, faces_lm68 in zip(detect_res, all_lm68):
            new_faces = []
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_face = (box, lm5, face_lm68, score)
                new_faces.append(new_face)
            all_detect_res.append(new_faces)

        detect_res = all_detect_res

        # print("split into super clips")

        tracks = multiple_tracking(detect_res)
        tuples = [(0, len(detect_res))] * len(tracks)

        # print("full_tracks", len(tracks))

        if len(tracks) == 0:
            tuples, tracks = find_longest(detect_res)

        data_storage = {}
        frame_boxes = {}
        super_clips = []

        for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
            # print(start, end)
            assert len(detect_res[start:end]) == len(track)

            super_clips.append(len(track))

            for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
                box, lm5, lm68 = face[:3]
                big_box = get_crop_box(shape, box, scale=0.5)

                top_left = big_box[:2][None, :]
                new_lm5 = lm5 - top_left
                new_lm68 = lm68 - top_left
                new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                info = (new_box, new_lm5, new_lm68, big_box)

                x1, y1, x2, y2 = big_box
                cropped = frames[frame_idx][y1:y2, x1:x2]
                base_key = f"{track_i}_{j}_"
                data_storage[f"{base_key}img"] = cropped
                data_storage[f"{base_key}ldm"] = info
                data_storage[f"{base_key}idx"] = frame_idx
                frame_boxes[frame_idx] = np.rint(box).astype(int)

        # print("sampling clips from super clips", super_clips)

        clips_for_video = []
        clip_size = cfg.clip_size
        pad_length = clip_size - 1

        for super_clip_idx, super_clip_size in enumerate(super_clips):
            inner_index = list(range(super_clip_size))
            if super_clip_size < clip_size: # padding
                # Create padding for pre_module and post_module carefully
                post_module = inner_index[1:-1][::-1] + inner_index
                pre_module = inner_index + inner_index[1:-1][::-1]

                if len(post_module) < pad_length:
                    post_module *= (pad_length // len(post_module) + 1)
                    
                post_module = post_module[:pad_length]
                    
                if len(pre_module) < pad_length:
                    pre_module *= (pad_length // len(pre_module) + 1)
                    
                pre_module = pre_module[-pad_length:]
                    
                assert len(post_module) == pad_length
                assert len(pre_module) == pad_length

                inner_index = pre_module + inner_index + post_module

            super_clip_size = len(inner_index)

            frame_range = [
                inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
            ]
            for indices in frame_range:
                clip = [(super_clip_idx, t) for t in indices]
                clips_for_video.append(clip)

        if len(clips_for_video) == 0:
            print(f'Skipping {input_path}: no viable clips')
            return None

        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

        with torch.no_grad():
            preds = []
            for c in range(0, len(clips_for_video), self.batch_size):
                clips = clips_for_video[c:min(len(clips_for_video), c+self.batch_size)]
                clip_batch = []
                for clip in clips:
                    images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
                    landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
                    frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
                    _, images_align = self.crop_align_func(landmarks, images)
                    images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
                    images = images.unsqueeze(0).sub(mean).div(std)

                    clip_batch.append(images)
                
                # Stack all tensors into one tensor (assuming they are all the same shape)
                clip_batch = torch.cat(clip_batch, dim=0)  # Concatenate along the batch dimension (dim=0)

                clip_batch = clip_batch.squeeze(1)

                # Pass the batch through the classifier
                outputs = self.model(clip_batch)
                
                # Store the output
                preds.extend(torch.sigmoid(outputs["final_output"].flatten()).cpu().tolist())

        return np.mean(preds)