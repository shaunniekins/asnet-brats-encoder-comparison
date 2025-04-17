# AS-Net Encoder Comparison for Brain Tumor Segmentation

This repository contains the code and experiments conducted for the research paper:

## Optimizing AS-Net for MRI Brain Tumor Segmentation: A Comparative Study of Different Encoders for Improved Segmentation Performance

The primary goal of this project was to investigate the impact of different convolutional neural network (CNN) encoders when integrated into the AS-Net (Attention Synergy Network) architecture for segmenting brain tumors in MRI scans from the BraTS 2023 dataset.

## Overview

The core components of this repository include:

* Implementations of the AS-Net model adapted to use various encoders (e.g., VGG16, MobileNetV3, EfficientNetV2).
* Scripts for training and evaluating these models on the BraTS 2023 dataset.
* Notebooks for model visualization and analysis.

## Encoders Compared

* VGG16 (Based on the original AS-Net study: *"AS-Net: Attention Synergy Network for Skin Lesion Segmentation"*)
* MobileNetV3 (Large and Small variants)
* EfficientNetV2 (B0, and B1 variants)

## Dataset

The models were trained and evaluated using the **Brain Tumor Segmentation (BraTS2023)** dataset. The specific dataset used was the ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData. Preprocessing steps involved converting the original NIfTI files into HDF5 format for efficient loading, organized into training, validation, and test splits.
