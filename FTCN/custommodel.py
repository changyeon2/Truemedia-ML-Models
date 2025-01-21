import argparse
import json
from time import perf_counter
from datetime import datetime
import os
import torch
import typing
import requests
import time  # Import the time module
import hashlib
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
import torch
import numpy as np

def generate_hashed_filename(url, original_filename):
    # Generate SHA-256 hash of the URL (32 characters long)
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    hashed_filename = f"{url_hash}_{original_filename}"
    
    return hashed_filename

def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"]) # "webm"
    )

def real_or_fake_threshold(prediction, threshold=0.5):
    if prediction == 0.5:
        return "NO_FACE"
    if prediction >= threshold:
        return "FAKE"
    else:
        return "REAL"

def download_file(input_path):
    """
    Download a file from a given URL and save it locally if input_path is a URL.
    If input_path is a local file path and the file exists, skip the download.

    :param input_path: The URL of the file to download or a local file path.
    :return: The local filepath to the downloaded or existing file.
    """
    # Check if input_path is a URL
    if input_path.startswith(('http://', 'https://')):
        # Extract filename from the URL
        # Splits the URL by '/' and get the last part
        filename = input_path.split('/')[-1]

        # Ensure the filename does not contain query parameters if present in the URL
        # Splits the filename by '?' and get the first part
        filename = filename.split('?')[0]

        # Define the local path where the file will be saved
        local_filepath = os.path.join('.', generate_hashed_filename(input_path, filename))

        # Check if file already exists locally
        if os.path.isfile(local_filepath):
            print(f"The file already exists locally: {local_filepath}")
            return local_filepath

        # Start timing the download
        start_time = time.time()

        # Send a GET request to the URL
        response = requests.get(input_path, stream=True)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Open the local file in write-binary mode
        with open(local_filepath, 'wb') as file:
            # Write the content of the response to the local file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # End timing the download
        end_time = time.time()

        # Calculate the download duration
        download_duration = end_time - start_time

        print(
            f"Downloaded file saved to {local_filepath} in {download_duration:.2f} seconds.")

    else:
        # Assume input_path is a local file path
        local_filepath = input_path
        # Check if the specified local file exists
        if not os.path.isfile(local_filepath):
            raise FileNotFoundError(f"No such file: '{local_filepath}'")
        print(f"Using existing file: {local_filepath}")

    return local_filepath

def delete_file(filename):
    """Deletes the specified file if it exists."""
    try:
        if os.path.isfile(filename):
            os.remove(filename)
            print(f"File '{filename}' has been deleted.")
        else:
            print(f"File '{filename}' does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")

class CustomModel:
    """Wrapper for a GenConvit model."""

    def __init__(self, checkpoint='checkpoints/ftcn_tt.pth', config='ftcn_tt.yaml', max_frames=768, batch_size=6, face_thres=0.75, bbox_thres=5000):
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.face_thres = face_thres
        self.bbox_thres = bbox_thres
        cfg.init_with_yaml()
        cfg.update_with_yaml(config)
        cfg.freeze()
        self.cfg = cfg

        model = PluginLoader.get_classifier(self.cfg.classifier_type)()
        model.load(checkpoint)
        model.cuda()
        model.eval()
        self.model = model

        print("The model is successfully loaded")

    def _predict(self,
                 input_file,
                 max_frames, 
                 batch_size
                 ):

        max_frames = max_frames if max_frames else self.max_frames
        batch_size = batch_size if batch_size else self.batch_size

        crop_align_func = FasterCropAlignXRay(self.cfg.imsize)

        print("detecting")
        detect_res, all_lm68, frames = detect_all(input_file, return_frames=True, max_size=max_frames, face_thres=self.face_thres, bbox_thres=self.bbox_thres)
        print("detect finished")

        print("number of frames: ", len(frames))

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

        print("split into super clips")

        tracks = multiple_tracking(detect_res)
        tuples = [(0, len(detect_res))] * len(tracks)

        print("full_tracks", len(tracks))

        if len(tracks) == 0:
            tuples, tracks = find_longest(detect_res)

        data_storage = {}
        frame_boxes = {}
        super_clips = []

        for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
            print(start, end)
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

        print("sampling clips from super clips", super_clips)

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
            print('Skipping: no viable clips')
            return 0.5

        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

        with torch.no_grad():
            preds = []
            for c in range(0, len(clips_for_video), batch_size):
                clips = clips_for_video[c:min(len(clips_for_video), c+batch_size)]
                clip_batch = []
                for clip in clips:
                    images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
                    landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
                    frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
                    _, images_align = crop_align_func(landmarks, images)
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

        y_val = np.mean(preds)
        print(f'df_probability: {y_val}')
        return y_val

    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        file_path = inputs.get('file_path', None)
        video_file = download_file(file_path)

        max_frames = inputs.get('max_frames', None) 
        batch_size = inputs.get('batch_size', None)

        if os.path.isfile(video_file):
            try:
                if is_video(video_file):
                    print(f"FTCN is being run.")
                    y_val = self._predict(
                        video_file,
                        max_frames,
                        batch_size
                    )

                    return {
                        "df_probability": y_val, "prediction": real_or_fake_threshold(y_val)}
                else:
                    print(
                        f"Invalid video file: {video_file}. Please provide a valid video file.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                raise # Raise the exception to allow the caller to handle it
            finally:
                # Ensure the video file is deleted after processing
                delete_file(video_file)
        else:
            print(f"The file {video_file} does not exist.")
        return

    @classmethod
    def fetch(cls) -> None:
        cls()


def main():
    """Entry point for interacting with this model via CLI."""
    start_time = perf_counter()
    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("-p", "--file_path",
                        help="The file path for the video file to predict on", required=True, default="https://www.evalai.org/ocasio.mp4")
    parser.add_argument("-c", "--config", default='ftcn_tt.yaml',
                        help="The config of ftcn")    
    parser.add_argument("-f", "--max_frames", type=int, default=768,
                        help="The number of frames to use for prediction")
    parser.add_argument("-b", "--batch_size", type=int, default=6,
                        help="Batch size for clip inference")

    args = parser.parse_args()

    if args.fetch:
        CustomModel.fetch()

    # Create an instance of CustomModel using the arguments
    model = CustomModel(
        config=args.config, max_frames=args.max_frames, batch_size=args.batch_size)

    # Create inputs dictionary for prediction
    inputs = {
        "file_path": args.file_path,
        "config": args.config,
        "max_frames": args.max_frames,
        "batch_size": args.batch_size,
    }
    # Call predict on the model instance with the specified arguments
    predictions = model.predict(inputs)

    # Optionally, print the predictions if you want to display them
    print(predictions)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
