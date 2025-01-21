
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose, ToTensor, Normalize

from model.ftcn import I3D8x8
from model.StyleGRU import StyleGRU

from PIL import Image
import pickle
import time
import requests

from time import perf_counter
from datetime import datetime
import typing

from preprocessing_utils import crop_face_from_video, set_result


def real_or_fake_thres(probability, threshold=0.2):
    if probability.item() == -1:
        return "NO_FACE"
    return "FAKE" if probability >= threshold else "REAL"

import re
def remove_extension(filename):
    return re.sub(r'\.(mp4|mov|webm|avi|mpg|mpeg)$', '', filename)

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
        local_filepath = os.path.join('.', filename)

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



_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

test_transform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


# FIXME : for sanic 
# before finetuning
# pth_path = {"content_pth": "/checkpoints/content/total_best.pth",
#             "style_pth": "/checkpoints/style/style_best.pth"}
# after finetuning
# pth_path = {"content_pth": "/checkpoints/content/finetuning.pth",
#             "style_pth": "/checkpoints/style/style_best.pth"}
# contents # 
pth_path = {"content_pth": "/checkpoints/content/true_best_1118.pth"}

# FIXME : for local
# pth_path = {"content_pth": "/workspace/StyleFlow/checkpoints/content/total_best.pth",
#             "style_pth": "/workspace/StyleFlow/checkpoints/style/style_best.pth"}
# pth_path = {"content_pth": "/workspace/StyleFlow/checkpoints/content/finetuning.pth",
#             "style_pth": "/workspace/StyleFlow/checkpoints/style/style_best.pth"}
# pth_path = {"content_pth": "/workspace/StyleFlow/checkpoints/content/contents.pth"}

# pth_path = {"content_pth": "/workspace/StyleFlow/checkpoints/content/true_best_1118.pth"}


data_root = "/data"
# FIXME : for Local
# data_root = "/workspace/StyleFlow/data"

class CustomModel:
    
    def __init__(self):
        """Initialize the model."""
        self.sigmoid = nn.Sigmoid()
        self.model = I3D8x8()
        # self.model_style = StyleGRU(feature_size=9216)
        checkpoint = torch.load(pth_path['content_pth'], map_location='cpu')
        # checkpoint_style = torch.load(pth_path['style_pth'], map_location='cpu')
        
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        else:
            pass
        
        self.model.load_state_dict(checkpoint, strict=False)
        # self.model_style.load_state_dict(checkpoint_style)
        self.model.to(_DEVICE)
        # self.model_style.to(_DEVICE)
        self.model.eval()
        # self.model_style.eval()
        print("The model is successfully loaded")
    
    def _predict(self, video_path, clip_size=32, max_frames=110):
   
        crop_face_from_video(video_path, clip_size=clip_size, max_frames=max_frames, data_root=data_root)
        # make_psp_features(video_path, clip_size=clip_size, max_frames=max_frames, data_root=data_root)

        video_name = remove_extension(video_path.split('/')[-1])
        image_root = os.path.join(data_root, video_name, 'crop_images')
        image_paths = [os.path.join(image_root, img) for img in sorted(os.listdir(image_root))]
        
        if len(image_paths) < 32:
            return torch.tensor(-1)
        
        num_frames = min(len(image_paths), max_frames)
        num_clips = num_frames // clip_size
        
        # psp_root = os.path.join(data_root, video_name, "psp")
        # psp_paths = [os.path.join(psp_root, p) for p in sorted(os.listdir(psp_root))]
        
        preds = []
        for idx in range(num_clips):
            start_idx = idx * 32
            end_idx = start_idx + 32
            
            # frames 
            clip = []
            for _idx in range(start_idx, end_idx, 1):
                with Image.open(image_paths[_idx]) as pil_img:
                    pil_img = pil_img.resize((224, 224))
                    pil_img = test_transform(pil_img)
                clip.append(pil_img)
          
            clip = torch.stack(clip, dim=1)
            
            clip = clip.unsqueeze(0)
            clip = clip.cuda(non_blocking=True)
            
            with torch.no_grad():
                ### ftcn ###
                outputs = self.model(clip)
            preds.append(outputs)
        
        # memory flush #
        clip = None
        
        return self.sigmoid(torch.mean(torch.stack(preds)))
                
    def predict(self,  inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        """Return a dict containing the completion of the given prompt.
        :param inputs: dict of inputs containing a prompt and optionally the max length
            of the completion to generate.
        :return: a dict containing the generated completion.
        """
        file_path = inputs.get('file_path', None)
        video_file = download_file(file_path)
        num_frames = inputs.get('num_frames', 32)
        max_frames = inputs.get('max_frames', 110)
        
        count = inputs.get('count', 0)
        
        # result_json = set_result()
        if os.path.join(video_file):
            try:
                if os.path.isfile(video_file) and video_file.endswith(tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])):
                    result = self._predict(video_file, clip_size=num_frames, max_frames=max_frames)
                    
                    return {"df_probability": result.item(), "prediction": real_or_fake_thres(result)}

                else:
                    print(f"Invalid video file: {video_file}. Please provide a valid video file.")
                    
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print(f"The file {video_file} does not exist.")
            
        return
                
    
    @classmethod
    def fetch(cls) -> None:
        cls()
    
    
def main():

    start_time = perf_counter()
    
    parser = argparse.ArgumentParser('StyleFlow arg script', add_help=False)
    # FIXME 
    parser.add_argument("-p", "--file_path", type = str,
                        help="The file path for the video file to predict on", required=True, default="https://www.evalai.org/ocasio.mp4")
    parser.add_argument("-f", "--num_frames", type=int, default=32,
                        help="StyleGRU default is 32")    
    parser.add_argument('--device', default='cuda',
                        help='device to use for inference')
    parser.add_argument('--max_frames', default=110, type=int,
                        help="max frames per video")
    parser.add_argument("--fetch", action="store_true")
    
    args = parser.parse_args()
    
    if args.fetch:
        CustomModel().fetch()
    
    model = CustomModel()
    
    inputs = {
        "file_path": args.file_path,
        "num_frames": args.num_frames,
        "max_frames": args.max_frames,
        "count": 0,
    }
    
    predictions = model.predict(inputs)
    
    print(predictions)    
    
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))
    
if __name__ == "__main__":
    main()