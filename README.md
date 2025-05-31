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

The models were trained and evaluated using the **Brain Tumor Segmentation (BraTS2023)** dataset. The specific dataset used was the ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData (it should be renamed into BraTS23_TrainingData). Preprocessing steps involved converting the original NIfTI files into HDF5 format for efficient loading, organized into training, validation, and test splits.

### Obtaining the Dataset

The BraTS 2023 dataset is available through the Synapse platform and requires registration. To simplify the data acquisition process, you can use the [brats-data-retriever](https://github.com/shaunniekins/brats-data-retriever) utility:

1. Clone the data retriever repository:

   ```bash
   git clone https://github.com/shaunniekins/brats-data-retriever
   cd brats-data-retriever
   ```

2. Set up a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install synapseclient synapseutils python-dotenv
   ```

3. Obtain a Synapse personal access token:
   * Register/login at [Synapse](https://www.synapse.org/)
   * Create a personal access token at [Synapse Personal Access Tokens](https://accounts.synapse.org/authenticated/personalaccesstokens)
   * Create a `.env` file with your token:

     ```
     SYNAPSE_AUTH_TOKEN=your_token_here
     DATASET_ID=syn53708249  # BraTS 2023 dataset ID
     ```

4. Run the data retrieval script:

   ```bash
   python main.py
   ```

5. Once downloaded, rename the dataset folder to `BraTS23_TrainingData` and place it in this project's root directory.

## System Requirements

* Python 3.8+
* CUDA-compatible GPU (recommended: at least 8GB VRAM)
* Sufficient disk space for dataset storage (~25GB for preprocessed data)
* 16GB+ RAM recommended

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shaunniekins/asnet-brats-encoder-comparison
cd asnet-brats-encoder-comparison
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, ensure you have the appropriate CUDA and cuDNN versions installed that are compatible with the TensorFlow version in requirements.txt.

## Workflow Overview

The complete workflow consists of these main steps:

1. Data preparation (splitting and preprocessing)
2. Model training for each encoder variant
3. Evaluation and inference
4. Result analysis

## How to Run

### 1. Prepare Dataset Splits

First, you'll need to generate train/validation/test splits from the BraTS23_TrainingData directory:

```bash
python3 create_splits.py --data_dir ./BraTS23_TrainingData --output_dir ./BraTS23_data_splits
```

Alternatively, you can run this step from the notebook `main.ipynb`.

### 2. Preprocess NIfTI Files to HDF5 Slices

Convert the NIfTI files into HDF5 slices for efficient training:

```bash
python3 preprocess_nifti_to_h5.py \
    --nifti_dir ./BraTS23_TrainingData \
    --split_dir ./BraTS23_data_splits \
    --output_dir ./BraTS23_preprocessed_h5_slices
```

This preprocessing step:

* Extracts 2D slices from 3D NIfTI volumes
* Normalizes the image data
* Creates binary segmentation masks
* Organizes the data into training, validation, and test sets

### 3. Train AS-Net Models

You can train different AS-Net variants by running the corresponding scripts:

* **VGG16 Backbone:**

  ```bash
  python3 asnet_vgg16_brats.py
  ```

* **MobileNetV3 Backbones:**

  ```bash
  python3 asnet_mobilenetv3_brats.py Small
  python3 asnet_mobilenetv3_brats.py Large
  ```

* **EfficientNetV2 Backbones:**

  ```bash
  python3 asnet_efficientnetv2_brats.py B0
  python3 asnet_efficientnetv2_brats.py B1
  ```

Or run all models sequentially from the notebook `main.ipynb`.

Each training script will:

* Load the preprocessed data
* Build the AS-Net model with the specified encoder
* Train the model for the specified number of epochs (default: 30)
* Save checkpoints and logs during training
* Evaluate the model on the validation set

### 4. Run Inference

After training, you can run inference on the test set using the `run_inference.py` script. Example:

```bash
python3 run_inference.py \
    --weights_path ./BraTS23_checkpoints/VGG16/VGG16_as_net_model_best.weights.h5 \
    --h5_data_dir ./BraTS23_preprocessed_h5_slices/test \
    --output_subdir results \
    --variant_name VGG16 \
    --img_height 224 \
    --img_width 224 \
    --batch_size_per_replica 32
```

Adjust the arguments as needed for other model variants. The inference script will:

* Load the trained model weights
* Run predictions on the test dataset
* Generate performance metrics
* Save visual examples of segmentation results
* Measure and report inference timing

### 5. Analyze Dataset Statistics

To analyze the dataset statistics (number of slices, patients, etc.):

```bash
python3 analyze_dataset_stats.py
```

Or run from the notebook `main.ipynb`.

### 6. Using Jupyter Notebooks

The repository includes two notebooks:

* `main.ipynb`: Main notebook for running the entire workflow
* `jupytext_converter.ipynb`: Utility for converting between notebooks and Python scripts

To run the notebooks:

```bash
jupyter lab
```

## Troubleshooting

* **GPU Memory Issues**: If you encounter out-of-memory errors, try reducing the batch size using the `--batch_size_per_replica` argument
* **Missing Files**: Make sure the BraTS23_TrainingData directory has the correct structure with NIfTI files named according to BraTS conventions
* **Checkpoint Loading Errors**: If model loading fails, check that the model architecture in the script matches the one used for training
