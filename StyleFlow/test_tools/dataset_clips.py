import os

from torch.utils.data import Dataset

def get_diff(array):
    array_ = array[1:, :]
    _array = array[:-1, :]
    return array_ - _array


def get_clip(iterable, clip_size=32):
    l = len(iterable)
    for ndx in range(0, l, clip_size):
        clip_paths = iterable[ndx:min(ndx+clip_size, l)]
        
        yield iterable[ndx:min(ndx+clip_size, l)]



class TestClips(Dataset):
    def __init__(
        self,
        video_path,
        frames_per_clip,
        grayscale=False,
        transform=None,
        max_frames_per_video=288,
        data_root="/data"
    ):
        self.frames_per_clip = frames_per_clip
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []
        self.clip_idxs = []
        
        self.latent_list = []
        
        video_name = video_path.split('/')[-1].replace('.mp4','')
        image_root = os.path.join(data_root, video_name, 'crop_image')
        self.paths = [os.path.join(image_root, img) for img in sorted(os.listdir(image_root))]
        
        self.num_frames = min(len(self.paths), max_frames_per_video)
        self.num_clips = self.num_frames // frames_per_clip
        
        # number of clips
        def __len__(self):
            return len(self.num_clips) 
        
        def __getitem__(self):
            pass
            
        
            
        
        
