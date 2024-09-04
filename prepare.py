import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns


def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df


def preprocess_data(tr_df, ts_df):
    valid_df, ts_df = train_test_split(
        ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])
    return tr_df, valid_df, ts_df


def create_generators(tr_df, valid_df, ts_df, batch_size=32, img_size=(299, 299)):
    _gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
    ts_gen = ImageDataGenerator(rescale=1/255)

    tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                      y_col='Class', batch_size=batch_size,
                                      target_size=img_size)

    valid_gen = _gen.flow_from_dataframe(valid_df, x_col='Class Path',
                                         y_col='Class', batch_size=batch_size,
                                         target_size=img_size)

    ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                        y_col='Class', batch_size=16,
                                        target_size=img_size, shuffle=False)

    return tr_gen, valid_gen, ts_gen


def plot_class_distribution(df, title='Count of images in each class'):
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(data=df, y=df['Class'])
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title, fontsize=20)
    ax.bar_label(ax.containers[0])
    plt.show()
