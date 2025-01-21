import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import optparse
import pandas as pd
import boto3
from io import BytesIO
    
def preprocess(input_file, filename, out_dir, bucket_name, folder_path, cfg, crop_align_func, max_clips, max_frames, face_thres=0.75, bbox_thres=5000):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

    num_channels = 3  # Number of channels (e.g., RGB)
    height, width = cfg.imsize, cfg.imsize  # Dimensions for each frame

    cache_file = f"{input_file}_{max_frames}.pth"

    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(input_file, max_size=max_frames, cvt=True)
        # print("detection result loaded from cache")
    else:
        # print("detecting")
        detect_res, all_lm68, frames = detect_all(input_file, return_frames=True, max_size=max_frames, face_thres=face_thres, bbox_thres=bbox_thres)
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

    preds = []

    if len(clips_for_video) == 0:
        print('Skipping: no viable clips')
        return

    # Grab clips equally throughout clips_for_video
    indices = np.linspace(0, len(clips_for_video) - 1, num=max_clips, dtype=int)
    clips_for_video = np.array(clips_for_video)[indices]

    # Initialize a tensor to hold all clips
    for k, clip in enumerate(clips_for_video):
        if k >= max_clips:  # Use >= to include max_clips if needed
            break
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        _, images_align = crop_align_func(landmarks, images)
        
        # Convert to tensor and normalize
        images_tensor = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)

        out_filename = f'{filename}_clip_{k}'

        # # Convert to video
        # tensor_to_video(images_tensor, out_dir, out_filename)
        # continue

        images_tensor = images_tensor.unsqueeze(0).sub(mean).div(std)  # Normalize

        if (not bucket_name and not folder_path):
            # Save the tensor locally
            torch.save(images_tensor, f'{out_dir}/{out_filename}.pt')
        else:
            s3 = boto3.client('s3')

            # Save to s3 # Create an in-memory buffer
            buffer = BytesIO()
            
            # Save the tensor to the buffer
            torch.save(images_tensor, buffer)
            
            # Move the cursor to the beginning of the buffer
            buffer.seek(0)

            # Upload the compressed buffer to S3
            s3.upload_fileobj(buffer, bucket_name, f'{folder_path}/{out_filename}.pt')

def tensor_to_video(images_tensor, out_dir, filename):
    tensors = images_tensor.cpu().numpy()

    # Ensure the tensor shape is [frames, height, width, channels]
    tensors = np.transpose(tensors, (1, 2, 3, 0))  # Convert to [frames, height, width, channels]

    # # Convert from RGB to BGR (OpenCV expects BGR)
    # tensor = (tensor * 255).astype(np.uint8)  # Scale to 0-255 for video format

    # Scale values to 0-255 if they are in [0, 1] range
    if tensors.max() <= 1.0:
        tensors = (tensors * 255).astype(np.uint8)
    else:
        tensors = tensors.astype(np.uint8)

    # Get video dimensions
    height, width = tensors.shape[1:3]
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    fps = 15
    video_writer = cv2.VideoWriter(f'{out_dir}/{filename}.mp4', fourcc, fps, (width, height))
    
    for frame in tensors:
        # Write each frame to the video file
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    # Release the VideoWriter
    video_writer.release()


def worker(gpu_id, is_local, filenames_per_gpu, args):
    bucket_name, folder_path = None, None

    if not is_local:
       bucket_name, folder_path = parse_s3_url(args.out_dir)
       print(f'Bucket Name: {bucket_name}')
       print(f'Folder Path: {folder_path}')

    torch.cuda.set_device(gpu_id)  # Set the current process to use the specified GPU

    cfg.init_with_yaml()
    cfg.update_with_yaml(args.config)
    cfg.freeze()

    filenames = filenames_per_gpu[gpu_id]
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    for i, filename in tqdm(enumerate(filenames), desc=f'GPU {gpu_id}'):
        print(filename)
        try:
            # Full path of the input file
            input_path = os.path.join(args.directory_path, filename)

            # Process the file
            preprocess(input_path, filename[:-4], args.out_dir, bucket_name, folder_path, cfg, crop_align_func, args.max_clips, args.max_frames, args.face_thres, args.bbox_thres)     

        except Exception as e:
            print(f"An error occurred with {filename}: {e}")

def args_parser():
    parser = optparse.OptionParser("Preprocess data for training.")
    parser.add_option("-d", "--dir", dest="directory_path", help="Data Directory.")
    parser.add_option("-o", "--output", dest="out_dir", help="Output path for Directory.")
    parser.add_option("-c", "--config", dest="config", help="Config", default='ftcn_tt.yaml')
    parser.add_option("--max_clips", dest="max_clips", type=int, help="Max clips to extract for each video", default=30)
    parser.add_option("--max_frames", dest="max_frames", type=int, help="Max frames to look for in a video", default=400)
    parser.add_option("--face_thres", dest="face_thres", type=int, help="Threshold for faces", default=0.75)
    parser.add_option("--bbox_thres", dest="bbox_thres", type=int, help="Threshold for boundry box area", default=5000)

    return parser.parse_args()

def parse_s3_url(s3_url):
    """Parse the S3 URL into bucket name and folder path."""
    if not s3_url.startswith('s3://'):
        raise ValueError("Invalid S3 URL: must start with 's3://'")
    
    path = s3_url[5:]  # Remove 's3://'
    bucket_name, *key_parts = path.split('/')  # Split by '/'
    folder_path = '/'.join(key_parts)  # Join the remaining parts to form the folder path
    
    return bucket_name, folder_path

if __name__ == "__main__":
    args, _ = args_parser()

    is_local = False if args.out_dir.startswith('s3://') else True 

    # Get all filenames already present in the output directory
    if (is_local):
        existing_filenames = set(os.listdir(args.out_dir))
    else:
        # Parse the S3 URL
        bucket_name, folder_path = parse_s3_url(args.out_dir)

        # Initialize the S3 client
        s3 = boto3.client('s3')

        # Get the list of existing filenames in the S3 folder
        existing_filenames = set()
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

        # Check if any files exist
        while True:
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):  # Skip folders
                        continue
                    filename = os.path.basename(key)  # Extract filename
                    existing_filenames.add(filename)

            # Check if there are more files to retrieve
            if response.get('IsTruncated'):  
                response = s3.list_objects_v2(
                    Bucket=bucket_name, Prefix=folder_path, ContinuationToken=response['NextContinuationToken']
                )
            else:
                break
                existing_filenames.add(filename)

    filenames = []
    try:
        files = os.listdir(args.directory_path)
        for file in files:
            # Check if the filename is a subset of any existing filename
            if any(file[:-4] in existing_file for existing_file in existing_filenames):
                print(f"Skipping {file}, already processed.")
                continue  # Skip to the next file

            if file.endswith(".mp4"):
                filenames.append(file)
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You do not have permission to access this directory.")

    if (len(filenames) != 0):
        # Define number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f'Number of gpus: {num_gpus}')
        print(f'Processing: {len(filenames)}')

        # Distribute work for real and fake filenames
        filenames_per_gpu = [filenames[i::num_gpus] for i in range(num_gpus)]

        # Create processes
        torch.multiprocessing.spawn(worker, args=(is_local, filenames_per_gpu, args) , nprocs=num_gpus, join=True)

        