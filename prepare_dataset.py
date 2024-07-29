# prepare_dataset.py

import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
height = 192
width = 256
channels = 3

# Directory paths
dataset_dir = 'brain_tumor_mri/'
train_dir = os.path.join(dataset_dir, 'Training')


def generate_data_paths(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if os.path.isdir(foldpath):
            filelist = os.listdir(foldpath)
            for file in filelist:
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})


# Create label encoder
def encode_labels(df):
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['labels'])
    return df, le


# Process image
def process_image(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        img = np.array(img.resize((width, height))).astype(np.float32)
        return img
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


# Load data in batches
def load_data_in_batches(df):
    images = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filepath)
                   for filepath in df['filepaths']]
        for future in as_completed(futures):
            image = future.result()
            if image is not None:
                images.append(image)
    return np.array(images), np.array(df['label_encoded'])


if __name__ == "__main__":
    # Generate paths
    train_df = generate_data_paths(train_dir)

    # Encode labels
    train_df, label_encoder = encode_labels(train_df)

    # Split data into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)

    # Load training and validation data
    print('Loading Brain Tumor MRI dataset...')
    Data_train, Label_train = load_data_in_batches(train_df)
    Data_val, Label_val = load_data_in_batches(val_df)
    print('Reading Brain Tumor MRI dataset finished')

    # Save the data
    save_dir = 'data/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'data_train'), Data_train)
    np.save(os.path.join(save_dir, 'label_train'), Label_train)
    np.save(os.path.join(save_dir, 'data_val'), Data_val)
    np.save(os.path.join(save_dir, 'label_val'), Label_val)

    # Save label encoder classes
    np.save(os.path.join(save_dir, 'label_classes'), label_encoder.classes_)
