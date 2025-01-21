import os
import torch
import optparse
from sklearn.model_selection import train_test_split
from utils.plugin_loader import PluginLoader
from config import config as cfg
from tqdm import tqdm
from trainer.ftcn_trainer import FTCNTrainer
import time
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import boto3
import pandas as pd

def load_data(data_dir, dataset_split, test_size=0.2, is_local=True, bucket_name=None, folder_path=None):
    """Load tensor paths and labels, then split them into train and validation sets."""
    data = []
    labels = []
    num_fake, num_real = 0, 0

    if (dataset_split):
        # Load the dataset split from the CSV file
        df = pd.read_csv(dataset_split)

        # Split based on the 'split' column
        train = df[df['split'] == 'train']
        val = df[df['split'] == 'val']

        # Extract the filenames and labels
        train_files = train['filename'].tolist()
        train_labels = train['label'].tolist()

        val_files = val['filename'].tolist()
        val_labels = val['label'].tolist()

        # Print the number of fake and real samples
        num_fake = sum(df['label'])
        num_real = len(df) - num_fake

        print(f'Fake: {num_fake}')
        print(f'Real: {num_real}')


        print(f'Train: {len(train_files)}')
        print(f'Val: {len(val_files)}')


        return train_files, val_files, train_labels, val_labels
    else:
        if (is_local):
            # Recursively traverse the directory structure
            for root, _, files in os.walk(data_dir):
                for filename in files:
                    if filename.endswith('.pt'):
                        file_path = os.path.join(root, filename)
                        # Check if 'fake' or 'real' is in the path
                        if 'fake' in file_path.lower():
                            data.append(file_path)  # label 1 = fake
                            labels.append(1)
                            num_fake += 1
                        elif 'real' in file_path.lower():
                            data.append(file_path)  # label 0 = real
                            labels.append(0)
                            num_real += 1
        else :
            # Initialize the S3 client
            s3 = boto3.client('s3')

            # Traverse s3   
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)

            # Collect all file paths, categorizing them by 'fake' or 'real' based on their path
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_path = obj['Key']
                        # Check if 'fake' or 'real' is in the path
                        if 'fake' in file_path.lower():
                            data.append(file_path)
                            labels.append(1)
                            num_fake += 1
                        elif 'real' in file_path.lower():
                            data.append(file_path)
                            labels.append(0)
                            num_real += 1


        print(f'Fake: {num_fake}')
        print(f'Real: {num_real}')
            

        # Split paths and labels into train and validation sets (80%-20%)
        train, val, train_labels, val_labels = train_test_split(
            data, labels, test_size=test_size, stratify=labels, random_state=42
        )

        print(f'Train: {len(train)}')
        print(f'Val: {len(val)}')

        return train, val, train_labels, val_labels

def save_acc_figures(train_accs, valid_accs, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Ensure the directory exists
    results_dir = f"result/{model_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot
    plot_filename = os.path.join(results_dir, f'acc_trend_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved as {plot_filename}")

def save_loss_figures(train_epoch_losses, valid_epoch_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_losses, label='Training Loss')
    plt.plot(valid_epoch_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()

    # Ensure the directory exists
    results_dir = f"result/{model_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot
    plot_filename = os.path.join(results_dir, f'loss_trend_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved as {plot_filename}")

def args_parser():
    parser = optparse.OptionParser("Train model.")
    parser.add_option("-d", "--dir", dest="dir", help="Training data path.")
    parser.add_option("-p", "--pretrained", dest="pretrained", help="Saved model to continue training.", default=None)
    parser.add_option("-c", "--config", dest="config", help="Config", default='ftcn_tt.yaml')
    parser.add_option("--split", dest="split", help="Dataset split", default=None)
    parser.add_option("--freeze", dest="freeze", action="store_true", help="Freeze Backbone", default=False)

    return parser.parse_args()

def parse_s3_url(s3_url):
        """Parse the S3 URL into bucket name and folder path."""
        if not s3_url.startswith('s3://'):
            raise ValueError("Invalid S3 URL: must start with 's3://'")
        
        path = s3_url[5:]  # Remove 's3://'
        bucket_name, *key_parts = path.split('/')  # Split by '/'
        folder_path = '/'.join(key_parts)  # Join the remaining parts to form the folder path
        
        return bucket_name, folder_path

def main(rank, world_size, model, model_dir, train, val, train_labels, val_labels, batch_size, test_batch_size, init_lr, total_epoch, lr_step, is_local=True, bucket_name=None, folder_path=None, freeze=False):
    print('Rank: ' + str(rank))

    model_name = f'ftcn_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'

    trainer = FTCNTrainer(model, model_name, model_dir, train, val, train_labels, val_labels, rank, world_size, batch_size, test_batch_size, init_lr, total_epoch, lr_step, is_local, bucket_name, folder_path, freeze)
    train_losses, val_losses, train_accuracies, val_accuracies = trainer.run()

    if rank == 0:
        save_loss_figures(train_losses, val_losses, model_name)
        save_acc_figures(train_accuracies, val_accuracies, model_name)

if __name__ == "__main__":
    # Parse command-line arguments
    args, _ = args_parser()

    # Load configuration
    cfg.init_with_yaml()
    cfg.update_with_yaml(args.config)
    cfg.freeze()

    print('Loading Model')
    # Load model
    model = PluginLoader.get_classifier(cfg.classifier_type)()
    if (args.pretrained):
        print('Loading Pretrained')
        model.load(args.pretrained)
    # model.cuda()

    world_size = torch.cuda.device_count()  # Number of GPUs available

    is_local = False if args.dir.startswith('s3://') else True
    print(f'Is Local: {is_local}')
    print(f'Freeze Backbone: {args.freeze}')

    bucket_name, folder_path = None, None

    if (not is_local):
        bucket_name, folder_path = parse_s3_url(args.dir)

    print(f'Bucket Name: {bucket_name}')
    print(f'Folder Path: {folder_path}')

    print(f'Loading data')
    # Split into train and validation sets
    train, val, train_labels, val_labels = load_data(args.dir, args.split, is_local=is_local, bucket_name=bucket_name, folder_path=folder_path)

    mp.spawn(main, args=(world_size, model, cfg.path.model_dir, train, val, train_labels, val_labels, int(cfg.trainer.default.batch_size), int(cfg.trainer.default.test_batch_size), float(cfg.trainer.default.init_lr), int(cfg.trainer.default.total_epoch), int(cfg.trainer.default.lr_step), is_local, bucket_name, folder_path, args.freeze), nprocs=world_size, join=True)

    print("Training complete!")

