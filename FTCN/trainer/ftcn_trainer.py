import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
from tqdm import tqdm
from utils.plugin_loader import PluginLoader
from trainer._base import TrainerBase
import numpy as np
import time
from trainer.s3_dataset import S3Dataset

# Custom collate function to filter out None values
def custom_collate(batch):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle case where all items were None
    # Unpack tensors and labels into separate lists
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels)

class FTCNTrainer(TrainerBase):
    def __init__(self, model, model_name, model_dir, train, val, train_labels, val_labels, rank, world_size, batch_size, test_batch_size, init_lr, total_epoch, lr_step, is_local=True, bucket_name=None, folder_path=None, freeze=False):
        # Initialize distributed training
        self.setup(rank, world_size)

        self.model_dir = model_dir
        self.model_name = model_name

        # Device setup
        self.device = torch.device(f'cuda:{rank}')

        # Freeze backbone
        if freeze:
            for name, param in model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

        self.model = DDP(model.to(self.device), device_ids=[rank])
        

        # Load parameters from config
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.init_lr = init_lr
        self.total_epoch = total_epoch

        # Optimizer, loss, and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step, gamma=0.1)

        self.is_local = is_local 

        # Get all filenames already present in the output directory
        self.bucket_name = bucket_name
        self.folder_path = folder_path

        train_dataset = S3Dataset(train, train_labels, self.device, self.is_local, self.bucket_name, self.folder_path)
        val_dataset = S3Dataset(val, val_labels, self.device, self.is_local, self.bucket_name, self.folder_path)

        # Distributed DataLoader with DistributedSampler
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank),
            collate_fn=custom_collate,
            num_workers=torch.cuda.device_count(),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.test_batch_size,
            sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank),
            collate_fn=custom_collate,
            num_workers=torch.cuda.device_count(),
        )

        # Mixed precision training scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Tracking metrics
        self.global_step = 0
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        

    def setup(self, rank, world_size):
        """Initialize the distributed process group."""
        dist.init_process_group(backend='gloo', init_method='tcp://localhost:12355', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup(self):
        """Cleanup the distributed process group."""
        dist.destroy_process_group()

    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for clips, labels in tqdm(self.train_loader, desc=f"(GPU {self.rank}) Training Epoch {epoch}"):
            clips = clips.squeeze(1)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(clips)

                loss = self.criterion(outputs['final_output'].squeeze(), labels.squeeze().float())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * clips.size(0)
            preds = (outputs['final_output'] > 0.5).float()
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)

            self.global_step += 1

            # Clean up
            del clips
            del labels
            del outputs
            del loss
            del preds
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Validation loop."""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for clips, labels in tqdm(self.val_loader, desc=f"(GPU {self.rank}) Validation"):
                clips = clips.squeeze(1)
                
                labels = labels.to(self.device)
                
                outputs = self.model(clips)
                loss = self.criterion(outputs['final_output'].squeeze(), labels.squeeze().float())

                running_loss += loss.item() * clips.size(0)
                preds = (outputs['final_output'] > 0.5).float()
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct / total
        return val_loss, val_acc

    def run(self):
        """Main training loop."""
        print(f'(GPU {self.rank}) Training')
        best_val_loss = float('inf')
        for epoch in range(1, self.total_epoch + 1):
            if self.rank == 0:
                print(f"(GPU {self.rank}) Epoch {epoch}/{self.total_epoch}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            if self.rank == 0:
                print(f"(GPU {self.rank}) Train Loss Epoch {epoch}: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"(GPU {self.rank}) Validation Loss Epoch {epoch}: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(epoch)

            torch.distributed.barrier()

            # self.scheduler.step()

        self.cleanup()
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

    def save_model(self, epoch):
        """Save model checkpoint."""
        if self.rank == 0:  # Ensure only rank 0 saves the model
            save_path = os.path.join(
                self.model_dir,  # Assuming model_dir is passed or initialized properly
                f'{self.model_name}_epoch_{epoch}.pth'
            )
            torch.save(self.model.module.network.state_dict(), save_path)  # Use .module with DDP
            print(f"Model saved at {save_path}")

