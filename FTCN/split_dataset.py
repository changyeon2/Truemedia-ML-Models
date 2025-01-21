import boto3
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_s3_url(s3_url):
    """Parse an S3 URL into bucket name and prefix."""
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    s3_url = s3_url[5:]
    bucket_name, _, prefix = s3_url.partition("/")
    return bucket_name, prefix


def list_s3_files(s3_client, bucket_name, prefix):
    """Retrieve all file paths under a given S3 prefix (relative to the bucket)."""
    files = []
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    while True:
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if not key.endswith('/'):  # Skip folder keys
                    files.append(key)  # Keep only the relative path
        if response.get('IsTruncated'):  # Handle pagination
            response = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=prefix, ContinuationToken=response['NextContinuationToken']
            )
        else:
            break

    return files


def create_single_csv(real_files, fake_files, train_ratio=0.8):
    """Split filenames into train and validation sets and save as a single CSV."""
    real_labels = [0] * len(real_files)  # Label all real files as 'real'
    fake_labels = [1] * len(fake_files)  # Label all fake files as 'fake'

    # Combine filenames and labels
    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    # Split into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, train_size=train_ratio, random_state=42, stratify=all_labels
    )

    # Create a single DataFrame with a "split" column
    train_df = pd.DataFrame({"filename": train_files, "label": train_labels, "split": "train"})
    val_df = pd.DataFrame({"filename": val_files, "label": val_labels, "split": "val"})

    # Combine train and validation sets into one DataFrame
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    combined_df.to_csv("dataset_splits.csv", index=False)
    print("Dataset split saved as dataset_splits.csv")


if __name__ == "__main__":
    # Configuration
    s3_url = "s3://ftcn-dataset/lin-finetune"
    real_folder = "real"
    fake_folder = "fake"
    train_ratio = 0.8

    # Parse S3 URL
    bucket_name, base_prefix = parse_s3_url(s3_url)

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Retrieve files (relative paths)
    real_files = list_s3_files(s3, bucket_name, os.path.join(base_prefix, real_folder))
    fake_files = list_s3_files(s3, bucket_name, os.path.join(base_prefix, fake_folder))

    print(f"Found {len(real_files)} real files and {len(fake_files)} fake files.")

    # Create a single CSV file with train/val splits
    create_single_csv(real_files, fake_files, train_ratio)
