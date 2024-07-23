# AS-Net for MRI Brain Tumor with VGG16, EfficientNetV2 and MobileNetV3

This is an implementation of AS-Net for mri brain tumor segmentation with various encoders. Based on the paper that can be found [here](https://doi.org/10.1016/j.eswa.2022.117112).

## Dataset

Download the dataset from [this link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).

## Setup and Process

1. Prepare the dataset:

   - Create a `dataset_brain_tumor_mri` directory.
   - Inside `dataset_brain_tumor_mri`, create the following subdirectories and add data from the downloaded dataset:
     - Training
     - Testing

2. Set up the environment:

   ```
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Create directories for checkpoints:

   ```
   mkdir data checkpoint output
   ```

4. Prepare the dataset:

   ```
   python3 prepare_dataset.py
   ```

5. Train the model:

   ```
   python3 train.py
   ```

6. Evaluate the model:

   ```
   python3 evaluate.py
   ```