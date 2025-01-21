from torch.utils.data import Dataset
import boto3
import torch
from io import BytesIO

class S3Dataset(Dataset):
    """Dataset that loads tensors from disk on-the-fly."""
    def __init__(self, data_paths, labels, device, is_local=True, bucket_name=None, folder_path=None):
        self.data_paths = data_paths  # Store file paths
        self.labels = labels              # Store labels
        self.device = device
        self.is_local = is_local 
        self.bucket_name = bucket_name
        self.folder_path = folder_path
        self.s3 = None


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        try:
            if (self.is_local):
                data = torch.load(self.data_paths[idx], map_location='cpu')
            else:
                if (self.s3 == None):
                    # Initialize the S3 client
                    self.s3 = boto3.client('s3')
                    
                obj = self.s3.get_object(Bucket=self.bucket_name, Key=self.data_paths[idx])
                data_stream = obj['Body'].read()

                # Load the tensor
                data = torch.load(BytesIO(data_stream), map_location='cpu')

            labels = torch.tensor(self.labels[idx], dtype=torch.float32).to('cpu')
        except Exception as e:
            print(f"Error loading {self.data_paths[idx]}: {e}")
            return None

        return data, labels