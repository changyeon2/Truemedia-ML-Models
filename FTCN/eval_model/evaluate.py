import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import optparse
import pandas as pd
from eval_model.eval_model import EvalModel
from datetime import datetime

def worker(gpu_id, args, filenames_per_gpu, labels_per_gpu):
    torch.cuda.set_device(gpu_id)  # Set the current process to use the specified GPU

    device = torch.device(f'cuda:{gpu_id}')

    output_csv = os.path.join(args.output_dir, f'predictions.csv')

    filenames = filenames_per_gpu[gpu_id]
    labels = labels_per_gpu[gpu_id]

    # Load the model
    model = EvalModel(device, config=args.config, weights=args.weights, max_frames=args.max_frames, batch_size=args.batch_size)

    # Initialize a list to collect rows
    rows = []

    for i, filename in tqdm(enumerate(filenames), desc=(f'GPU {gpu_id}')):
        try:
            # Perform inference and create a row
            likelihood = model.predict(f'{args.directory_path}/{labels[i].lower()}/{filename}')
            # If likelihood > 0.5, classify as Fake (1), otherwise classify as Real (0)
            predicted = (likelihood > 0.5).astype(int) if likelihood is not None else None
            rows.append({'filename': filename, 'truth': labels[i], 'likelihood': likelihood, 'predicted': predicted})
        except Exception as e:
            print(f"An error occurred with {filename}: {e}")

        # Save every 10 rows to the CSV
        if i % 10 == 0 and rows:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
            rows = []  # Clear the rows list after saving

    # Save any remaining rows to the CSV
    if rows:
        temp_df = pd.DataFrame(rows)
        temp_df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))

def args_parser():
    parser = optparse.OptionParser("Train model.")
    parser.add_option("-d", "--dir", dest="directory_path", help="Data Directory.")
    parser.add_option("-o", "--output", dest="output_dir", help="Output Directory.")
    parser.add_option("-w", "--weights", dest="weights", help="Path to pretrained weights", default="checkpoints/ftcn_tt.pth")
    parser.add_option("-c", "--config", dest="config", help="Config", default='ftcn_tt.yaml')
    parser.add_option("--max-frames", dest="max_frames", type=int, help="Max frames to look for in a video", default=400)
    parser.add_option("--batch-size", dest="batch_size", type=int, help="Batch size for inference", default=1)

    return parser.parse_args()

def calulate_metrics(output_dir):
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df = pd.read_csv(csv_path)

    # Define cutoff threshold for likelihood to classify as positive (Fake)
    threshold = 0.5

    # Drop all rows that don't have a likelihood value
    df = df.dropna(subset=['likelihood'])

    print(f'Predicted Real: {(df["predicted"] == 0).sum()}')
    print(f'Predicted Fake: {(df["predicted"] == 1).sum()}')

    # Map truth column to binary values for comparison
    # Assuming 'Real' is 0 and 'Fake' is 1
    df['truth'] = df['truth'].map({'Real': 0, 'Fake': 1})

    # Calculate the evaluation metrics
    accuracy = accuracy_score(df['truth'], df['predicted'])
    f1 = f1_score(df['truth'], df['predicted'])
    precision = precision_score(df['truth'], df['predicted'])
    recall = recall_score(df['truth'], df['predicted'])

    # Confusion matrix to get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(df['truth'], df['predicted']).ravel()

    # Calculate additional rates
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Calculate AUC (using likelihoods as predicted probabilities)
    auc = roc_auc_score(df['truth'], df['likelihood'])

    # Display results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'True Negative Rate: {tn_rate:.2f}')
    print(f'False Positive Rate: {fp_rate:.2f}')
    print(f'False Negative Rate: {fn_rate:.2f}')
    print(f'AUC: {auc:.2f}')

    # Save metrics to a csv file
    metrics = pd.DataFrame({
        'Accuracy': [accuracy],
        'F1 Score': [f1],
        'Precision': [precision],
        'Recall': [recall],
        'True Negative Rate': [tn_rate],
        'False Positive Rate': [fp_rate],
        'False Negative Rate': [fn_rate],
        'AUC': [auc]
    })

    metrics_path = os.path.join(output_dir, 'metrics.csv')

    metrics.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    args, _ = args_parser()

    # Refactor the output directory in eval_model folder
    args.output_dir = os.path.join('eval_model/', args.output_dir) 
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'predictions.csv')

    try:
        df = pd.read_csv(csv_path)
        print('CSV loaded')
    except FileNotFoundError:
        print("The file was not found. Creating new file...")
        df = pd.DataFrame(columns=['filename', 'truth', 'likelihood', 'predicted'])
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Load dataset
    real_filenames = []
    fake_filenames = []

    print(args.directory_path)
    try:
        files = os.listdir(args.directory_path + '/real')
        for file in files:
            if file.endswith(('.mp4', '.png', '.jpeg', '.jpg','.mp3')) and not file in df['filename'].values:
                real_filenames.append(file)

        files = os.listdir(args.directory_path + '/fake')
        for file in files:
            if file.endswith(('.mp4', '.png', '.jpeg', '.jpg', '.mp3')) and not file in df['filename'].values:
                fake_filenames.append(file)
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You do not have permission to access this directory.")

    print("Real: " + str(len(real_filenames)))
    print("Fake: " + str(len(fake_filenames)))

    # Concatenate the filenames
    filenames = real_filenames + fake_filenames

    # Create the labels list by repeating 'Real' for real_filenames and 'Fake' for fake_filenames
    labels = ['Real'] * len(real_filenames) + ['Fake'] * len(fake_filenames)

    assert len(filenames) == len(labels)

    if (len(filenames) != 0):
        # Define number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f'Number of gpus: {num_gpus}')

        # Distribute work for real and fake filenames
        filenames_per_gpu = [filenames[i::num_gpus] for i in range(num_gpus)]
        labels_per_gpu = [labels[i::num_gpus] for i in range(num_gpus)]

        # Create processes
        torch.multiprocessing.spawn(worker, args=(args, filenames_per_gpu, labels_per_gpu) , nprocs=num_gpus, join=True)

    # Calculate metrics
    calulate_metrics(args.output_dir)
        