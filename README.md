# AS-Net Encoder Comparison for Brain Tumor Segmentation

This repository contains the code and experiments conducted for the research paper:

## Optimizing AS-Net for MRI Brain Tumor Segmentation: A Comparative Study of Different Encoders for Improved Segmentation Performance

The primary goal of this project was to investigate the impact of different convolutional neural network (CNN) encoders when integrated into the AS-Net (Attention Synergy Network) architecture for segmenting brain tumors in MRI scans from the BraTS 2020 dataset.

## Overview

The core components of this repository include:

* Implementations of the AS-Net model adapted to use various encoders (e.g., MobileNetV3, EfficientNetV2).
* Scripts for training and evaluating these models on the BraTS 2020 dataset.
* Notebooks for model visualization and analysis.

## Encoders Compared

* MobileNetV3 (Large and Small variants)
* EfficientNetV2 (B0 and B1 variants)
* VGG16 (Based on the original AS-Net study: *"AS-Net: Attention Synergy Network for Skin Lesion Segmentation"*)

## Dataset

The models were trained and evaluated using the **Brain Tumor Segmentation (BraTS2020)** dataset available on Kaggle: [https://www.kaggle.com/datasets/awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). Preprocessing steps involved converting the original NIfTI files into HDF5 format for efficient loading.
