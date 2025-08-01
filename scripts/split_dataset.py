import os
import shutil
import random
from pathlib import Path

# Set paths
combined_dir = Path(r"E:\lung_cnn_diagnosis\combined_dataset")
output_dir = Path(r"E:\lung_cnn_diagnosis\train_val_test_split")

# Class labels
categories = ['cancerous', 'non_cancerous']

# Split ratios
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Seed for reproducibility
random.seed(42)

def split_and_copy():
    for category in categories:
        # List images
        images = list((combined_dir / category).glob("*.jpg"))
        random.shuffle(images)

        total = len(images)
        train_end = int(split_ratio['train'] * total)
        val_end = train_end + int(split_ratio['val'] * total)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Copy files
        for split_name, file_list in splits.items():
            split_folder = output_dir / split_name / category
            split_folder.mkdir(parents=True, exist_ok=True)

            for img_path in file_list:
                dst_path = split_folder / img_path.name
                shutil.copy(img_path, dst_path)

            print(f"[{split_name.upper()}] {category}: {len(file_list)} images")

if __name__ == "__main__":
    split_and_copy()
