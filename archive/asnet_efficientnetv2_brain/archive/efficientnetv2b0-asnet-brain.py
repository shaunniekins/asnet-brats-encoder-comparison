# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Imports and Setup

# %pip install numpy keras matplotlib scikit-learn pandas tensorflow[and-cuda] nvidia-cudnn-cu12 h5py -q

import os
import gc
import time
import math
import numpy as np
import tensorflow as tf
from keras import Model, Input, backend
# Import EfficientNetV2 models and preprocessing
from keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3 # Add more variants if needed
from keras.applications import efficientnet_v2 as efficientnet_v2_preprocessing # Use alias
from keras.layers import (
    Conv2D,
    BatchNormalization,
    # MaxPooling2D, # Keep if needed by custom modules, maybe not by main arch
    Activation,
    UpSampling2D,
    concatenate,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    Add, # Import Add layer
)
from keras.losses import Loss
from keras.metrics import Metric
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tensorflow.keras import mixed_precision
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix # Removed sklearn, use Keras metrics

# Limit TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = INFO filtered, 2 = WARNING filtered, 3 = ERROR filtered

# ## Constants

# +
# --- Model Variant ---
# Select the EfficientNetV2 variant by uncommenting ONE of the following lines:
EFFICIENTNET_VARIANT = 'EfficientNetV2B0'
# EFFICIENTNET_VARIANT = 'EfficientNetV2B1'
# EFFICIENTNET_VARIANT = 'EfficientNetV2B2' # Example: add B2 if desired
# EFFICIENTNET_VARIANT = 'EfficientNetV2B3' # Example: add B3 if desired

# --- Input Dimensions based on Variant ---
if EFFICIENTNET_VARIANT == 'EfficientNetV2B0':
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B1':
    IMG_HEIGHT = 240
    IMG_WIDTH = 240
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B2':
    IMG_HEIGHT = 260
    IMG_WIDTH = 260
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B3':
    IMG_HEIGHT = 300
    IMG_WIDTH = 300
else:
    # Default or raise error for unsupported variants
    print(f"Warning: Using default size 224x224 for unlisted variant {EFFICIENTNET_VARIANT}")
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

INPUT_CHANNELS = 3 # For EfficientNetV2 input

# --- Data Loading Constants ---
# Select modalities (Indices based on BraTS documentation: 0:T1, 1:T1ce, 2:T2, 3:FLAIR)
MODALITY_INDICES = [1, 3]  # Use T1ce and FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES) # Should be 2
# Define which loaded modality corresponds to which RGB channel for EfficientNetV2 input
# Example: [T1ce, FLAIR, T1ce] -> Index 0, Index 1, Index 0 from the loaded modalities
RGB_MAPPING_INDICES = [0, 1, 0] # Map T1ce to R and B, FLAIR to G

# --- Training Constants ---
# Adjust batch size based on GPU memory and image size
BATCH_SIZE = 4      # Start low, adjust based on variant and GPU memory
LEARNING_RATE = 1e-4 # Initial learning rate (Keep based on previous success)
NUM_EPOCHS = 30     # Number of training epochs (or until early stopping)
BUFFER_SIZE = 300   # Shuffle buffer size (adjust based on memory)
THRESHOLD = 0.5     # Segmentation threshold for binary metrics and visualization

# --- Loss Weights ---
# Use weights from successful VGG16 run
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5, 'dice_weight': 0.5, 'class_weight': 100.0} # Tunable loss weights

# --- Paths ---
# Parameterize paths based on the EfficientNetV2 variant
VARIANT_SUFFIX = EFFICIENTNET_VARIANT.lower() # e.g., "efficientnetv2b0"
CHECKPOINT_DIR = f"./{VARIANT_SUFFIX}-checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{VARIANT_SUFFIX}_as_net_model.weights.h5"
CHECKPOINT_BEST_PATH = f"{CHECKPOINT_DIR}/{VARIANT_SUFFIX}_as_net_model_best.weights.h5"
OUTPUT_DIR = f"{VARIANT_SUFFIX}-output"
COMPLETION_FILE = f"{VARIANT_SUFFIX}-asnet-finished-training.txt"
DATASET_PATH = "brats2020-training-data/" # Make sure this points to the directory containing 'BraTS20 Training Metadata.csv' and 'content/data'
METADATA_FILE = os.path.join(DATASET_PATH, "BraTS20 Training Metadata.csv")
H5_DATA_DIR = os.path.join(DATASET_PATH, "BraTS2020_training_data/content/data") # Directory containing the .h5 slice files

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "examples"), exist_ok=True)
# -

# ## GPU Configuration and Mixed Precision

# +
print("--- GPU Configuration ---")
# Configure memory growth to prevent GPU memory overflow upfront
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        # MirroredStrategy for multi-GPU training
        if len(gpus) > 1:
             strategy = tf.distribute.MirroredStrategy()
             print(f"Running on {strategy.num_replicas_in_sync} GPU(s) using MirroredStrategy.")
        else:
             strategy = tf.distribute.get_strategy() # Default for single GPU
             print("Running on single GPU (default strategy).")

    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}. Falling back to default strategy.")
        strategy = tf.distribute.get_strategy() # Fallback to default (CPU or single GPU)
        print("Running on CPU or single GPU (default strategy).")
else:
    strategy = tf.distribute.get_strategy() # Default strategy (CPU)
    print("No GPU detected. Running on CPU.")
print("Number of replicas in sync:", strategy.num_replicas_in_sync)
print("Global Batch Size (per replica * num replicas):", BATCH_SIZE * strategy.num_replicas_in_sync)


print("\n--- Mixed Precision Configuration ---")
# Use mixed precision to reduce memory usage and potentially speed up training on compatible GPUs
# policy = mixed_precision.Policy('mixed_float16') # Uncomment for mixed precision
policy = mixed_precision.Policy('float32') # Using float32 as requested
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy set to: {policy.name}")
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# +
# Configure JIT compilation (Optional, can sometimes improve performance)
# tf.config.optimizer.set_jit(True)
# print("JIT compilation enabled.")
# -

# ## Define AS-Net Model Architecture with EfficientNetV2 Encoder

# +
# Helper function to find layer names (useful during development)
def find_layer_names(model, substring):
    names = []
    for layer in model.layers:
        if substring in layer.name:
            names.append(layer.name)
    return names

def AS_Net(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), variant=EFFICIENTNET_VARIANT):
    """Defines the AS-Net model with an EfficientNetV2 encoder."""
    # Create an explicit input tensor with a name
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Preprocess input for EfficientNetV2 (typically 0 to 1 scaling, handled by the layer/function)
    # The preprocessing function handles the specific scaling needs.
    preprocessed_inputs = efficientnet_v2_preprocessing.preprocess_input(inputs)

    # Load EfficientNetV2 backbone pre-trained on ImageNet
    if variant == 'EfficientNetV2B0':
        print("Loading EfficientNetV2B0 encoder...")
        base_model = EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs
        )
        # Updated skip connection layers based on actual EfficientNetV2B0 architecture
        skip_layer_names = [
            'block1a_project_activation', # ~112x112 (for 224 input)
            'block2b_add',                # ~56x56
            'block3b_add',                # ~28x28
            'block5e_add',                # ~14x14
            'top_activation'              # ~7x7 (Bottleneck)
        ]
        fine_tune_start_block = 4 # Start unfreezing from block 4 or 5
    elif variant == 'EfficientNetV2B1':
        print("Loading EfficientNetV2B1 encoder...")
        base_model = EfficientNetV2B1(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs
        )
        # Updated skip connection layers based on actual EfficientNetV2B1 architecture
        skip_layer_names = [
            'block1a_project_activation', # ~120x120 (for 240 input)
            'block2c_add',                # ~60x60
            'block3c_add',                # ~30x30
            'block5f_add',                # ~15x15
            'top_activation'              # ~8x8 (Bottleneck)
        ]
        fine_tune_start_block = 4 # Start unfreezing from block 4 or 5
    elif variant == 'EfficientNetV2B2':
        print("Loading EfficientNetV2B2 encoder...")
        base_model = EfficientNetV2B2(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs
        )
        # Updated skip connection layers based on actual EfficientNetV2B2 architecture
        skip_layer_names = [
            'block1a_project_activation', # ~130x130 (for 260 input)
            'block2c_add',                # ~65x65
            'block3c_add',                # ~33x33
            'block5g_add',                # ~17x17
            'top_activation'              # ~9x9 (Bottleneck)
        ]
        fine_tune_start_block = 4
    elif variant == 'EfficientNetV2B3':
        print("Loading EfficientNetV2B3 encoder...")
        base_model = EfficientNetV2B3(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs
        )
        # Updated skip connection layers based on actual EfficientNetV2B3 architecture
        skip_layer_names = [
            'block1a_project_activation', # ~150x150 (for 300 input)
            'block2d_add',                # ~75x75
            'block3d_add',                # ~38x38
            'block5j_add',                # ~19x19
            'top_activation'              # ~10x10 (Bottleneck)
        ]
        fine_tune_start_block = 4
    else:
        raise ValueError(f"Unsupported EfficientNetV2 variant: {variant}")

    # --- Fine-tuning ---
    base_model.trainable = True # Ensure base is trainable before selective freezing
    freeze_until_block = f'block{fine_tune_start_block}'
    print(f"Fine-tuning: Freezing layers up to block '{freeze_until_block}'...")
    layer_found = False
    for layer in base_model.layers:
        if freeze_until_block in layer.name:
            layer_found = True
            print(f"Reached block '{freeze_until_block}' at layer: {layer.name}. Unfreezing subsequent layers.")
        if not layer_found:
            layer.trainable = False
        else:
            # Ensure Batch Norm layers remain frozen if desired (often recommended during fine-tuning)
            # Or keep them trainable if BN stats need adjustment for medical images
            # if isinstance(layer, BatchNormalization):
            #     layer.trainable = False
            # else:
            layer.trainable = True # Unfreeze layers from the target block onwards

    if not layer_found:
        print(f"Warning: Block '{freeze_until_block}' not found for freezing. Fine-tuning all layers.")
        for layer in base_model.layers: # Ensure all are trainable if block wasn't found
             layer.trainable = True

    # Count trainable/non-trainable params
    trainable_count = sum([tf.keras.backend.count_params(w) for w in base_model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in base_model.non_trainable_weights])
    print(f'Base Model - Trainable params: {trainable_count:,} | Non-trainable params: {non_trainable_count:,}')
    # --- End Fine-tuning ---


    # Extract feature maps from EfficientNetV2 encoder stages
    encoder_outputs = []
    print("Extracting skip connections from layers:")
    for name in skip_layer_names:
        try:
            layer_output = base_model.get_layer(name).output
            encoder_outputs.append(layer_output)
            print(f" - {name}: Shape {layer_output.shape}")
        except ValueError as e:
            print(f"\nERROR: Could not find layer '{name}' in {variant}. Error: {e}")
            print("Available layers (first 50):")
            for i, layer in enumerate(base_model.layers):
                 # Fix the error for printing layer information
                 if hasattr(layer, 'output_shape'):
                     print(f"  {i}: {layer.name} - Output Shape: {layer.output_shape}")
                 else:
                     print(f"  {i}: {layer.name}")
                 if i > 50: break # Print only first few layers
            raise ValueError(f"Layer {name} not found. Check skip_layer_names for {variant}.")

    # Unpack encoder outputs (adjust based on the number of skip connections)
    if len(encoder_outputs) != 5:
        raise ValueError(f"Expected 5 skip connections, but got {len(encoder_outputs)} for {variant}")
    output1, output2, output3, output4, bottleneck = encoder_outputs
    # Shapes for B0: (112, 112), (56, 56), (28, 28), (14, 14), (7, 7)
    # Shapes for B1: (120, 120), (60, 60), (30, 30), (15, 15), (8, 8)
    # Shapes for B2: (130, 130), (65, 65), (33, 33), (17, 17), (9, 9)
    # Shapes for B3: (150, 150), (75, 75), (38, 38), (19, 19), (10, 10)


    # --- Decoder with SAM, CAM, and Synergy ---
    # Adapts the VGG16/MobileNetV3 decoder structure
    # Handles potential odd dimensions from some EfficientNet variants via bilinear upsampling

    # Decoder Stage 1: Bottleneck (H/32) -> H/16
    print(f"\nDecoder Stage 1 (H/32 -> H/16)")
    up4 = UpSampling2D(size=(2, 2), interpolation="bilinear", name='up_bottleneck')(bottleneck)
    # Ensure shapes match before concat (resize if needed, though UpSampling should handle it)
    # up4 = tf.image.resize(up4, tf.shape(output4)[1:3], method='bilinear') # Optional: Explicit resize
    print(f"  Upsampled bottleneck shape: {up4.shape}")
    print(f"  Target skip connection (output4) shape: {output4.shape}")
    merge4 = concatenate([output4, up4], axis=-1, name='merge4')
    filters4 = merge4.shape[-1]
    print(f"  Concatenated shape (H/16): {merge4.shape}, Filters: {filters4}")
    SAM4 = SAM(filters=filters4, name='sam4')(merge4)
    CAM4 = CAM(filters=filters4, name='cam4')(merge4)
    print(f"  SAM4/CAM4 output shape: {SAM4.shape}")

    # Decoder Stage 2: H/16 -> H/8
    print(f"Decoder Stage 2 (H/16 -> H/8)")
    up_sam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    print(f"  Upsampled SAM4/CAM4 shape: {up_sam4.shape}")
    print(f"  Target skip connection (output3) shape: {output3.shape}")
    merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
    merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
    filters3 = merge31.shape[-1]
    print(f"  Concatenated shape (H/8): {merge31.shape}, Filters: {filters3}")
    SAM3 = SAM(filters=filters3, name='sam3')(merge31)
    CAM3 = CAM(filters=filters3, name='cam3')(merge32)
    print(f"  SAM3/CAM3 output shape: {SAM3.shape}")

    # Decoder Stage 3: H/8 -> H/4
    print(f"Decoder Stage 3 (H/8 -> H/4)")
    up_sam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    print(f"  Upsampled SAM3/CAM3 shape: {up_sam3.shape}")
    print(f"  Target skip connection (output2) shape: {output2.shape}")
    merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
    merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
    filters2 = merge21.shape[-1]
    print(f"  Concatenated shape (H/4): {merge21.shape}, Filters: {filters2}")
    SAM2 = SAM(filters=filters2, name='sam2')(merge21)
    CAM2 = CAM(filters=filters2, name='cam2')(merge22)
    print(f"  SAM2/CAM2 output shape: {SAM2.shape}")

    # Decoder Stage 4: H/4 -> H/2
    print(f"Decoder Stage 4 (H/4 -> H/2)")
    up_sam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    print(f"  Upsampled SAM2/CAM2 shape: {up_sam2.shape}")
    print(f"  Target skip connection (output1) shape: {output1.shape}")
    merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
    merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
    filters1 = merge11.shape[-1]
    print(f"  Concatenated shape (H/2): {merge11.shape}, Filters: {filters1}")
    SAM1 = SAM(filters=filters1, name='sam1')(merge11)
    CAM1 = CAM(filters=filters1, name='cam1')(merge12)
    print(f"  SAM1/CAM1 output shape: {SAM1.shape}")

    # Final Upsampling Stage: H/2 -> H (Full Resolution)
    print(f"Decoder Stage 5 (H/2 -> H)")
    final_up_sam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
    final_up_cam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
    print(f"  Final upsampled SAM/CAM shape: {final_up_sam.shape}")


    # Synergy module to combine final SAM and CAM outputs
    synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam]) # Input shapes: (H, W, C_out_synergy) -> Output: (H, W, 1)
    print(f"  Synergy output shape: {synergy_output.shape}")

    # Final 1x1 convolution for segmentation map (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid", name='final_output', dtype='float32')(synergy_output)
    print(f"  Final output shape: {output.shape}")

    # Create the model with explicit input and output
    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{variant}')

    # Clean up memory
    gc.collect()

    return model


# --- SAM Module ---
# (Keep SAM, CAM, Synergy classes exactly as they were in the VGG16/MobileNetV3 version)
class SAM(Model):
    """Spatial Attention Module"""
    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        # Output channels reduction (ensure reasonable number of channels)
        self.out_channels = max(16, filters // 8 if filters > 128 else filters // 4) # Adjust reduction based on input filters

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3') # F'(X) path
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # F''(X) path (shortcut)

        # Pooling and Upsampling for spatial attention map generation
        # Use MaxPooling2D from Keras layers
        from keras.layers import MaxPooling2D
        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.upsample1 = UpSampling2D((2, 2), interpolation="bilinear", name='upsample1')
        self.W1 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W1') # Output 1 channel attention map

        self.pool2 = MaxPooling2D((4, 4), name='pool2')
        self.upsample2 = UpSampling2D((4, 4), interpolation="bilinear", name='upsample2')
        self.W2 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W2') # Output 1 channel attention map

        self.add_attention = Add(name='add_attention')
        self.multiply = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs))) # Main feature path F'(X)
        out2 = self.conv4(inputs) # Shortcut path F''(X)

        # Parallel attention branches (on shortcut path F''(X))
        pool1 = self.pool1(out2)
        up1 = self.upsample1(pool1)
        # Explicit resize to match shortcut path dimension precisely if needed (e.g., due to padding)
        up1 = tf.image.resize(up1, size=tf.shape(out2)[1:3], method="bilinear")
        att1 = self.W1(up1) # S1 (shape: B, H, W, 1)

        pool2 = self.pool2(out2)
        up2 = self.upsample2(pool2)
        up2 = tf.image.resize(up2, size=tf.shape(out2)[1:3], method="bilinear")
        att2 = self.W2(up2) # S2 (shape: B, H, W, 1)

        # Combine attention weights (S)
        attention_map = self.add_attention([att1, att2]) # S = S1 + S2 (shape: B, H, W, 1)

        # Apply attention: Multiply main path by attention weights and add shortcut
        # Y = F'(X) * S + F''(X)
        # Attention map (B,H,W,1) is broadcasted across channels of out1 (B,H,W,C_out)
        attended_features = self.multiply([out1, attention_map]) # F'(X) * S
        y = self.add_residual([attended_features, out2]) # Add shortcut F''(X)
        return y

    def get_config(self):
        config = super(SAM, self).get_config()
        config.update({"filters": self.filters})
        return config

# --- CAM Module ---
class CAM(Model):
    """Channel Attention Module"""
    def __init__(self, filters, reduction_ratio=16, name='cam', **kwargs):
        super(CAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        # Output channels reduction (ensure reasonable number of channels)
        self.out_channels = max(16, filters // 8 if filters > 128 else filters // 4) # Adjust reduction based on input filters
        self.reduction_ratio = reduction_ratio

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3') # F'(X) path
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # F''(X) path (shortcut)

        # Channel attention mechanism
        self.gpool = GlobalAveragePooling2D(name='global_avg_pool', keepdims=True) # Keep dims for broadcasting
        reduced_channels = max(1, self.out_channels // self.reduction_ratio) # Ensure at least 1 channel
        self.fc1 = Dense(reduced_channels, activation="relu", use_bias=False, dtype=compute_dtype, name='fc1')
        self.fc2 = Dense(self.out_channels, activation="sigmoid", use_bias=False, dtype=compute_dtype, name='fc2')

        self.multiply = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs))) # Main feature path F'(X)
        out2 = self.conv4(inputs) # Shortcut path F''(X)

        # Calculate channel attention weights (C) from shortcut path F''(X)
        pooled = self.gpool(out2) # Global Average Pooling -> (Batch, 1, 1, C_out)
        fc1_out = self.fc1(pooled)
        channel_attention_weights = self.fc2(fc1_out) # Shape: (Batch, 1, 1, C_out)

        # Apply attention: Multiply main path by channel attention weights and add shortcut
        # Y = F'(X) * C + F''(X)
        recalibrated_features = self.multiply([out1, channel_attention_weights]) # F'(X) * C
        y = self.add_residual([recalibrated_features, out2]) # Add shortcut F''(X)
        return y

    def get_config(self):
        config = super(CAM, self).get_config()
        config.update({"filters": self.filters, "reduction_ratio": self.reduction_ratio})
        return config

# --- Synergy Module ---
class Synergy(Model):
    """Combines SAM and CAM outputs with learnable weights."""
    def __init__(self, alpha_init=0.5, beta_init=0.5, name='synergy', **kwargs):
        super(Synergy, self).__init__(name=name, **kwargs)
        # Use tf.Variable for learnable weights, ensure they are float32 for stability
        self.alpha = tf.Variable(alpha_init, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(beta_init, trainable=True, name="beta", dtype=tf.float32)

        # Conv + BN after weighted sum. Output should have 1 channel for final segmentation.
        compute_dtype = mixed_precision.global_policy().compute_dtype
        # Ensure output is 1 channel for the final sigmoid in AS_Net
        self.conv = Conv2D(1, 1, padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv')
        self.bn = BatchNormalization(name='bn')
        self.add = Add(name='add_weighted') # Use Add layer

    def call(self, inputs):
        sam_features, cam_features = inputs # Expecting tuple/list: (SAM_output, CAM_output)

        # Cast learnable weights to the compute dtype of the inputs for multiplication
        compute_dtype = sam_features.dtype
        alpha_casted = tf.cast(self.alpha, compute_dtype)
        beta_casted = tf.cast(self.beta, compute_dtype)

        # Weighted sum: alpha * SAM_output + beta * CAM_output
        weighted_sum = self.add([alpha_casted * sam_features, beta_casted * cam_features])

        # Apply Conv -> BN
        convolved = self.conv(weighted_sum)
        bn_out = self.bn(convolved)
        # No activation here, final sigmoid is in the main model output layer
        return bn_out # Output shape: (B, H, W, 1)

    def get_config(self):
        config = super(Synergy, self).get_config()
        # Store initial values, actual values are saved in weights
        config.update({"alpha_init": 0.5, "beta_init": 0.5})
        return config
# -

# ## Loss Functions

# +
# Keep Loss functions (DiceLoss, WBCE, CombinedLoss) exactly as they were
# Ensure they handle potential dtype casting correctly if using mixed precision

class DiceLoss(Loss):
    """Computes the Dice Loss."""
    def __init__(self, smooth=1e-6, name='dice_loss', **kwargs):
        super(DiceLoss, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coef

    def get_config(self):
        config = super(DiceLoss, self).get_config()
        config.update({"smooth": self.smooth})
        return config


class WBCE(Loss):
    """Weighted Binary Cross-Entropy Loss."""
    def __init__(self, weight=1.0, name='weighted_bce_loss', **kwargs):
        super(WBCE, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs)
        self.weight = tf.cast(weight, tf.float32) # Store weight as float32

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Match prediction dtype
        epsilon_ = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
        logits = tf.math.log(y_pred / (1.0 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=tf.cast(self.weight, logits.dtype) # Cast weight to logits dtype
        )
        # Framework handles reduction based on 'sum_over_batch_size'
        return loss

    def get_config(self):
        config = super(WBCE, self).get_config()
        config.update({"weight": self.weight.numpy()}) # Store numpy value
        return config

# Combined Loss (Dice + Weighted BCE)
class CombinedLoss(Loss):
    """Combines Dice Loss and Weighted Binary Cross-Entropy."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, class_weight=1.0, name='combined_loss', **kwargs):
        super(CombinedLoss, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wbce = WBCE(weight=class_weight)
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        bce_loss_val = self.wbce(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)
        # Combine losses with weights; framework handles reduction
        combined = (self.bce_weight * bce_loss_val) + (self.dice_weight * dice_loss_val)
        return combined

    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "class_weight": self.wbce.weight.numpy()
        })
        return config
# -

# ## Custom Metrics

# +
# Keep Metrics (DiceCoefficient, IoU) exactly as they were
# Ensure they handle potential dtype casting and use tf operations

# Dice Coefficient Metric
class DiceCoefficient(Metric):
    """Computes the Dice Coefficient metric."""
    def __init__(self, threshold=THRESHOLD, smooth=1e-6, name='dice_coefficient', dtype=None):
        super(DiceCoefficient, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros', dtype=tf.float32)
        self.union_sum = self.add_weight(name='union_sum', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_binary, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        pred_sum = tf.reduce_sum(y_pred_f)
        true_sum = tf.reduce_sum(y_true_f)
        self.intersection_sum.assign_add(intersection)
        self.union_sum.assign_add(true_sum + pred_sum)

    def result(self):
        dice = (2.0 * self.intersection_sum + self.smooth) / (self.union_sum + self.smooth)
        return tf.cast(dice, self._dtype) if self._dtype else dice

    def reset_state(self):
        self.intersection_sum.assign(0.0)
        self.union_sum.assign(0.0)

    def get_config(self):
        config = super(DiceCoefficient, self).get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


# IoU Metric (Jaccard)
class IoU(Metric):
    """Computes the Intersection over Union (IoU) or Jaccard Index."""
    def __init__(self, threshold=THRESHOLD, smooth=1e-6, name='iou', dtype=None):
        super(IoU, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros', dtype=tf.float32)
        self.union_sum = self.add_weight(name='union_sum', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_binary, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        true_sum = tf.reduce_sum(y_true_f)
        pred_sum = tf.reduce_sum(y_pred_f)
        union = true_sum + pred_sum - intersection
        self.intersection_sum.assign_add(intersection)
        self.union_sum.assign_add(union)

    def result(self):
        iou = (self.intersection_sum + self.smooth) / (self.union_sum + self.smooth)
        return tf.cast(iou, self._dtype) if self._dtype else iou

    def reset_state(self):
        self.intersection_sum.assign(0.0)
        self.union_sum.assign(0.0)

    def get_config(self):
        config = super(IoU, self).get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config
# -

# ## Data Preparation for BraTS (Generic - Preprocessing done in model)

# +
def prepare_brats_data_gpu(
    metadata_file=METADATA_FILE,
    h5_dir=H5_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Use updated target size from constants
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    modality_indices=MODALITY_INDICES,
    rgb_mapping_indices=RGB_MAPPING_INDICES,
    num_modalities_loaded=NUM_MODALITIES_LOADED,
    input_channels=INPUT_CHANNELS, # Should be 3
    validation_split=0.2,
    random_seed=42,
):
    """
    Prepares the BraTS 2020 dataset from H5 slices using tf.data.
    Outputs Z-scored RGB images; model-specific preprocessing happens in the model.
    """
    print("--- Setting up Data Pipeline ---")
    print(f"Target image size: {target_size}")
    # ... (rest of the path checking and metadata loading is the same)
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"H5 data directory not found at {h5_dir}")

    df = pd.read_csv(metadata_file)
    df['full_path'] = df['slice_path'].apply(lambda x: os.path.join(h5_dir, os.path.basename(x)))
    df = df[df['full_path'].apply(os.path.exists)].copy()

    if df.empty:
        raise ValueError(f"No valid H5 files found based on metadata in {h5_dir}. Check paths.")
    print(f"Found {len(df)} existing H5 files referenced in metadata.")

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - validation_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_files = train_df["full_path"].tolist()
    val_files = val_df["full_path"].tolist()

    # Function to parse H5 file (remains the same)
    def parse_h5_file(file_path):
        def _parse_h5(path_tensor):
            path = path_tensor.numpy().decode("utf-8")
            try:
                with h5py.File(path, "r") as hf:
                    image_data = hf["image"][()] # (H, W, 4) float64
                    mask_data = hf["mask"][()]   # (H, W, 3) uint8
                    selected_modalities = image_data[:, :, modality_indices].astype(np.float32)
                    binary_mask = np.logical_or.reduce((mask_data[:, :, 0] > 0,
                                                        mask_data[:, :, 1] > 0,
                                                        mask_data[:, :, 2] > 0)).astype(np.float32)
                    return selected_modalities, binary_mask
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                original_h, original_w = 240, 240
                return (np.zeros((original_h, original_w, num_modalities_loaded), dtype=np.float32),
                        np.zeros((original_h, original_w), dtype=np.float32))

        image, mask = tf.py_function(_parse_h5, [file_path], [tf.float32, tf.float32])
        original_h, original_w = 240, 240 # Assume original BraTS size
        image.set_shape([original_h, original_w, num_modalities_loaded])
        mask.set_shape([original_h, original_w])
        return image, mask

    # Function to preprocess: Z-score, RGB map, Resize, Finalize Mask
    def preprocess(image, mask):
        # --- Normalization (Z-score per modality) ---
        normalized_channels = []
        for i in range(num_modalities_loaded):
            channel = image[:, :, i]
            mean = tf.reduce_mean(channel)
            std = tf.math.reduce_std(channel)
            normalized_channel = (channel - mean) / (std + 1e-8) # Add epsilon
            normalized_channels.append(normalized_channel)
        image_normalized = tf.stack(normalized_channels, axis=-1)

        # --- RGB Mapping ---
        rgb_channels = [image_normalized[:, :, idx] for idx in rgb_mapping_indices]
        image_rgb = tf.stack(rgb_channels, axis=-1) # Shape: (H_orig, W_orig, 3)

        # --- Resizing ---
        image_resized = tf.image.resize(image_rgb, target_size, method='bilinear')
        mask_expanded = tf.expand_dims(mask, axis=-1)
        mask_resized = tf.image.resize(mask_expanded, target_size, method='nearest')
        mask_final = tf.squeeze(mask_resized, axis=-1)
        mask_final = tf.cast(mask_final > 0.5, tf.float32) # Ensure binary
        mask_final = tf.expand_dims(mask_final, axis=-1) # Add channel dim (H, W, 1)

        # --- Final Image Format ---
        # Ensure image is float32. Model-specific preprocessing (like EfficientNetV2's)
        # will be applied INSIDE the model definition.
        image_final = tf.cast(image_resized, tf.float32)

        # --- Set Final Shapes ---
        image_final.set_shape([target_size[0], target_size[1], input_channels])
        mask_final.set_shape([target_size[0], target_size[1], 1])

        return image_final, mask_final

    # --- Create tf.data Datasets ---
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_files)

    # --- Apply Transformations ---
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = (
        train_dataset.with_options(options)
        .shuffle(buffer_size) # Shuffle file paths
        .map(parse_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_dataset.with_options(options)
        # No shuffle for validation
        .map(parse_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Data pipeline created successfully.")
    print("Train Dataset Element Spec:", train_dataset.element_spec)
    print("Validation Dataset Element Spec:", val_dataset.element_spec)

    return train_dataset, val_dataset

# Function to visualize samples from the dataset (shows Z-scored input)
def visualize_dataset_samples(dataset, num_samples=3, output_dir=OUTPUT_DIR):
    """Visualizes samples (Z-scored inputs) from the dataset and saves the plot."""
    print("--- Visualizing Dataset Samples ---")
    try:
        plt.figure(figsize=(15, 5 * num_samples))
        plot_count = 0
        for images, masks in dataset.take(1): # Take one batch
            num_in_batch = images.shape[0]
            print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
            print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}")
            print(f"Image value range (Z-scored): Min={tf.reduce_min(images):.4f}, Max={tf.reduce_max(images):.4f}")
            print(f"Mask value range: Min={tf.reduce_min(masks):.4f}, Max={tf.reduce_max(masks):.4f}")

            for i in range(min(num_samples, num_in_batch)):
                plot_count += 1
                img_zscored = images[i].numpy()
                mask = masks[i].numpy().squeeze()

                titles = ["Input R (Z)", "Input G (Z)", "Input B (Z)", "RGB Input (Z-scored)", "Ground Truth Mask"]
                channels_to_plot = [img_zscored[:, :, 0], img_zscored[:, :, 1], img_zscored[:, :, 2], img_zscored, mask]

                for j, item in enumerate(channels_to_plot):
                    ax = plt.subplot(num_samples, len(titles), i * len(titles) + j + 1)
                    ax.set_title(titles[j])

                    if j < 4: # Z-scored image data
                        # Clip Z-score data approx to [0,1] for display using percentiles
                        p_low, p_high = np.percentile(item.flatten(), [2, 98])
                        item_disp = np.clip((item - p_low) / (p_high - p_low + 1e-8), 0, 1)
                        if np.isnan(item_disp).any() or np.isinf(item_disp).any(): item_disp = np.nan_to_num(item_disp)
                        plt.imshow(item_disp, cmap='gray' if item.ndim==2 else None)
                    else: # Mask
                        if np.isnan(item).any() or np.isinf(item).any(): item = np.nan_to_num(item)
                        plt.imshow(item, cmap='jet', vmin=0, vmax=1)

                    plt.axis("off")

        plt.tight_layout(pad=0.5)
        save_path = os.path.join(output_dir, "dataset_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dataset visualization saved to {save_path}")

    except Exception as e:
        print(f"Error during dataset visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
# -

# ## Training Callbacks

# +
# Keep ConciseProgressCallback and lr_step_decay exactly as they were

class ConciseProgressCallback(tf.keras.callbacks.Callback):
    """Logs progress concisely and performs garbage collection."""
    def __init__(self, log_frequency=1):
        super().__init__()
        self.log_frequency = log_frequency
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        if (epoch + 1) % self.log_frequency == 0:
             print(f"\n--- Epoch {epoch + 1}/{self.params['epochs']} ---")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start_time
        if (epoch + 1) % self.log_frequency == 0:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - {metrics_str}")
            try: # Monitor Synergy weights
                synergy_layer = self.model.get_layer('synergy')
                alpha_val = synergy_layer.alpha.numpy()
                beta_val = synergy_layer.beta.numpy()
                print(f"    Synergy weights - alpha: {alpha_val:.4f}, beta: {beta_val:.4f}")
            except Exception as e: pass # Layer might not exist
        gc.collect() # Force garbage collection

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n--- Training Finished ---")
        print(f"Total training time: {total_time:.2f} seconds")


# Learning Rate Scheduler Function
def lr_step_decay(epoch, lr):
    """Applies step decay to the learning rate."""
    initial_lr = LEARNING_RATE
    drop = 0.5
    epochs_drop = 10
    new_lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    final_lr = max(new_lr, 1e-7) # Minimum LR
    return final_lr
# -

# ## Training Function

def train_model_distributed(
    dataset_func=prepare_brats_data_gpu,
    model_func=AS_Net,
    strategy=strategy, # Pass the distribution strategy
    checkpoint_path=CHECKPOINT_PATH, # Use variant-specific path
    checkpoint_best_path=CHECKPOINT_BEST_PATH, # Use variant-specific path
    output_dir=OUTPUT_DIR, # Use variant-specific path
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    input_channels=INPUT_CHANNELS,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    combined_loss_weights=COMBINED_LOSS_WEIGHTS,
    metrics=['binary_accuracy',
             DiceCoefficient(name='dice_coef'),
             IoU(name='iou'),
             tf.keras.metrics.Precision(thresholds=THRESHOLD, name="precision"),
             tf.keras.metrics.Recall(thresholds=THRESHOLD, name="recall")],
    efficientnet_variant=EFFICIENTNET_VARIANT # Pass variant to model function
):
    """Trains the AS-Net EfficientNetV2 model using tf.distribute.Strategy."""

    global_batch_size = batch_size * strategy.num_replicas_in_sync
    print(f"--- Starting Training ({VARIANT_SUFFIX}) ---") # Include variant
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Target Image Size: {img_height}x{img_width}")
    print(f"Epochs: {num_epochs}")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Loss configuration: {combined_loss_weights}")

    # 1. Create Datasets within Strategy Scope
    print("Preparing datasets...")
    # Pass global_batch_size and correct target size
    train_dataset, val_dataset = dataset_func(
        batch_size=global_batch_size,
        target_size=(img_height, img_width)
    )
    print("Datasets prepared.")

    # Optionally visualize dataset samples before training
    visualize_dataset_samples(train_dataset, num_samples=3, output_dir=output_dir)

    # 2. Build and Compile Model within Strategy Scope
    with strategy.scope():
        print("Building model...")
        # Pass variant and input size to the model function
        model = model_func(
            input_size=(img_height, img_width, input_channels),
            variant=efficientnet_variant
        )
        print("Model built.")

        print("Compiling model...")
        print("Using Combined WBCE + Dice Loss")
        loss_instance = CombinedLoss(
             bce_weight=combined_loss_weights['bce_weight'],
             dice_weight=combined_loss_weights['dice_weight'],
             class_weight=combined_loss_weights['class_weight']
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if mixed_precision.global_policy().name == 'mixed_float16':
             optimizer = mixed_precision.LossScaleOptimizer(optimizer)
             print("Loss scaling applied for mixed precision.")

        model.compile(
            optimizer=optimizer,
            loss=loss_instance,
            metrics=metrics
        )
        print("Model compiled.")
        model.summary(line_length=120)

    # Check for checkpoint resume
    latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
    initial_epoch = 0
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        try:
            with strategy.scope(): # Load weights within scope
                model.load_weights(latest_checkpoint).expect_partial()
            # Simple epoch extraction (adjust regex/split if needed)
            try:
                epoch_str = latest_checkpoint.split('_')[-1].split('.')[0] # Try common patterns
                if not epoch_str.isdigit(): epoch_str = latest_checkpoint.split('.')[-2].split('-')[-1] # Another pattern
                if epoch_str.isdigit(): initial_epoch = int(epoch_str)
                print(f"Successfully loaded weights. Starting from epoch {initial_epoch + 1}")
            except:
                 print("Warning: Could not determine epoch from checkpoint name. Starting from epoch 0.")
                 initial_epoch = 0
        except Exception as load_err:
            print(f"Error loading weights: {load_err}. Starting from scratch.")
            initial_epoch = 0
    else:
        print("No checkpoint found, starting training from scratch.")

    # 3. Define Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_best_path, save_weights_only=True, monitor='val_dice_coef',
            mode='max', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, save_freq='epoch', verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=0),
        ConciseProgressCallback(log_frequency=1),
        # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=1) # Optional
    ]

    # 4. Train the Model
    epochs_to_run = num_epochs - initial_epoch
    print(f"Starting training loop for {epochs_to_run} epochs (Total planned: {num_epochs})...")
    if epochs_to_run <= 0:
        print("Training already completed based on initial_epoch. Skipping fit.")
        # If training completed, maybe load history? For now, just return None.
        # Try to load history if it exists
        hist_csv_file = os.path.join(output_dir, 'training_history.csv')
        if os.path.exists(hist_csv_file):
            print(f"Loading existing training history from {hist_csv_file}")
            history_df = pd.read_csv(hist_csv_file)
            # Convert DataFrame back to a history-like dictionary object if needed
            # For simplicity, we can skip plotting if training didn't run this time
            history = None # Indicate no new history generated
        else:
             history = None
    else:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=0 # Use custom callback
        )

    # 5. Save Training History (only if training ran)
    if history and history.history:
        try:
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = os.path.join(output_dir, 'training_history.csv')
            hist_df.to_csv(hist_csv_file, index=False)
            print(f"Training history saved to {hist_csv_file}")
            plot_training_history(history, output_dir)
        except Exception as e:
            print(f"Error saving/plotting training history: {e}")
    elif history is None and epochs_to_run <=0:
        print("Skipping history saving/plotting as training was already complete.")
    else:
        print("Warning: No training history generated or history object is empty.")


    # Cleanup
    print(f"Cleaning up resources after training ({VARIANT_SUFFIX})...")
    del train_dataset, val_dataset
    gc.collect()
    # Keep model instance for potential evaluation return
    print("Cleaned up datasets.")

    return history, model
# -

# ## Plot Training History

def plot_training_history(history, output_dir=OUTPUT_DIR):
    """Plots training & validation loss and metrics and saves the plot."""
    # This function remains the same as in the MobileNetV3 version
    print("--- Plotting Training History ---")
    try:
        history_dict = history.history
        if not history_dict:
             print("History object is empty. Skipping plotting.")
             return

        epochs = range(1, len(history_dict['loss']) + 1)
        metrics_to_plot = {'loss': 'Loss'}
        for key in history_dict.keys():
            if key.startswith('val_'): continue
            if key == 'loss': continue
            if key == 'dice_coef': metrics_to_plot['dice_coef'] = 'Dice Coef'
            elif key == 'iou': metrics_to_plot['iou'] = 'IoU'
            elif key == 'binary_accuracy': metrics_to_plot['binary_accuracy'] = 'Accuracy'
            elif key == 'precision': metrics_to_plot['precision'] = 'Precision'
            elif key == 'recall': metrics_to_plot['recall'] = 'Recall'
            elif f'val_{key}' in history_dict:
                 metrics_to_plot[key] = key.replace('_', ' ').title()

        num_plots = len(metrics_to_plot)
        if num_plots <= 1:
            print("Warning: Only 'loss' metric found. Plotting loss only.")
            if num_plots == 0: return

        plt.figure(figsize=(max(12, 6 * num_plots), 5))

        plot_index = 1
        for metric, title in metrics_to_plot.items():
            plt.subplot(1, num_plots, plot_index)
            val_metric = f'val_{metric}'
            if metric in history_dict:
                plt.plot(epochs, history_dict[metric], 'bo-', label=f'Training {title}')
            if val_metric in history_dict:
                plt.plot(epochs, history_dict[val_metric], 'ro-', label=f'Validation {title}')
            plt.title(f'{title}')
            plt.xlabel('Epoch')
            if metric != 'loss' and val_metric in history_dict: plt.legend()
            elif metric in history_dict and not val_metric in history_dict: plt.legend() # Show training legend if val missing
            if metric != 'loss': plt.ylim([0, 1.05])
            plt.grid(True)
            plot_index += 1

        plt.suptitle(f'AS-Net {EFFICIENTNET_VARIANT} Training History', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, "training_history_plots.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history plots saved to {save_path}")

    except Exception as e:
        print(f"Error plotting training history: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()

# ## Evaluation Function

def evaluate_model(
    model=None, # Pass the trained model or load from checkpoint
    checkpoint_best_path=CHECKPOINT_BEST_PATH, # Use variant path
    checkpoint_path = CHECKPOINT_PATH, # Last epoch path
    output_folder=OUTPUT_DIR, # Use variant path
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    input_channels=INPUT_CHANNELS,
    batch_size=BATCH_SIZE,
    dataset_func=prepare_brats_data_gpu,
    threshold=THRESHOLD,
    num_examples_to_save=5,
    loss_config=COMBINED_LOSS_WEIGHTS,
    efficientnet_variant=EFFICIENTNET_VARIANT # Pass variant
):
    """Evaluates the trained AS-Net EfficientNetV2 model."""
    print(f"\n--- Starting Model Evaluation ({VARIANT_SUFFIX}) ---")
    evaluation_results = None

    try:
        # 1. Load Validation Data
        global_eval_batch_size = batch_size * strategy.num_replicas_in_sync
        print(f"Loading validation data with batch size: {global_eval_batch_size}...")
        _, val_dataset = dataset_func(
            batch_size=global_eval_batch_size,
            target_size=(img_height, img_width), # Use correct size
            validation_split=0.2
        )
        print("Validation dataset loaded.")

        # 2. Load or Use Provided Model
        model_eval = None
        if model is None:
             print(f"Loading model weights from best checkpoint: {checkpoint_best_path}")
             checkpoint_to_load = None
             if os.path.exists(checkpoint_best_path):
                 checkpoint_to_load = checkpoint_best_path
             else:
                 print(f"Warning: Best checkpoint not found at {checkpoint_best_path}.")
                 last_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
                 # Check if index file exists for the checkpoint
                 if last_checkpoint and os.path.exists(last_checkpoint + ".index"):
                      print(f"Attempting to load last epoch checkpoint: {last_checkpoint}")
                      checkpoint_to_load = last_checkpoint
                 else:
                     print(f"Error: No suitable checkpoint found in {os.path.dirname(checkpoint_path)}. Cannot evaluate.")
                     return None

             print("Rebuilding model architecture for evaluation...")
             with strategy.scope():
                 model_eval = AS_Net(
                     input_size=(img_height, img_width, input_channels),
                     variant=efficientnet_variant # Pass variant
                 )
                 print("Compiling evaluation model...")
                 loss_instance = CombinedLoss(**loss_config)
                 optimizer = tf.keras.optimizers.Adam()
                 if mixed_precision.global_policy().name == 'mixed_float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                 model_eval.compile(
                      optimizer=optimizer,
                      loss=loss_instance,
                      metrics=['binary_accuracy',
                               DiceCoefficient(name='dice_coef'),
                               IoU(name='iou'),
                               tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
                               tf.keras.metrics.Recall(thresholds=threshold, name="recall")]
                 )
                 print(f"Loading weights from {checkpoint_to_load}...")
                 load_status = model_eval.load_weights(checkpoint_to_load)
                 load_status.expect_partial() # Allow optimizer state mismatch
                 print(f"Successfully loaded weights into new model instance.")
        else:
            print("Using provided trained model instance for evaluation.")
            model_eval = model
            if not model_eval.optimizer: # Compile if needed
                 print("Compiling the provided model for evaluation...")
                 with strategy.scope():
                      loss_instance = CombinedLoss(**loss_config)
                      optimizer = tf.keras.optimizers.Adam()
                      if mixed_precision.global_policy().name == 'mixed_float16': optimizer = mixed_precision.LossScaleOptimizer(optimizer)
                      model_eval.compile(
                           optimizer=optimizer, loss=loss_instance,
                           metrics=['binary_accuracy', DiceCoefficient(name='dice_coef'), IoU(name='iou'),
                                    tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
                                    tf.keras.metrics.Recall(thresholds=threshold, name="recall")]
                      )
                      print("Provided model compiled.")

        # 3. Evaluate
        print("Evaluating model on validation set...")
        evaluation_results = model_eval.evaluate(val_dataset, verbose=1, return_dict=True)
        print("\nKeras Evaluation Results:")
        for name, value in evaluation_results.items():
            print(f"- {name}: {value:.4f}")

        # 4. Calculate F1 Score
        precision_val = evaluation_results.get('precision', 0.0)
        recall_val = evaluation_results.get('recall', 0.0)
        f1_val = 0.0
        if (precision_val + recall_val) > 1e-7:
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        evaluation_results['f1_score'] = f1_val
        print(f"- f1_score: {f1_val:.4f} (calculated)")

        # 5. Save Performance Metrics
        try:
            perf_file_path = os.path.join(output_folder, "performances.txt")
            with open(perf_file_path, "w") as file_perf:
                file_perf.write(f"Evaluation Metrics ({VARIANT_SUFFIX}):\n")
                file_perf.write("------------------------------------\n")
                # Iterate through ordered keys if possible, or just the dict
                metric_order = ['loss', 'binary_accuracy', 'dice_coef', 'iou', 'precision', 'recall', 'f1_score']
                for name in metric_order:
                     if name in evaluation_results:
                          file_perf.write(f"- {name.replace('_', ' ').title()}: {evaluation_results[name]:.4f}\n")
                # Add any other metrics not in the predefined order
                for name, value in evaluation_results.items():
                     if name not in metric_order:
                          file_perf.write(f"- {name.replace('_', ' ').title()}: {value:.4f}\n")

            print(f"Evaluation results saved to {perf_file_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")

        # 6. Save Prediction Examples
        print("\nGenerating prediction examples...")
        save_prediction_examples(model_eval, val_dataset, output_folder, num_examples=num_examples_to_save, threshold=threshold)

        print(f"--- Evaluation Finished ({VARIANT_SUFFIX}) ---")
        return evaluation_results

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("Cleaning up resources after evaluation...")
        if 'val_dataset' in locals(): del val_dataset
        # Don't delete model_eval if it's the same as the passed 'model'
        if 'model_eval' in locals() and model_eval is not model: del model_eval
        gc.collect()
        print("Cleaned up evaluation resources.")
# -

# ## Save Prediction Examples

def save_prediction_examples(model, dataset, output_folder, num_examples=5, threshold=THRESHOLD):
    """Saves example predictions with inputs (Z-scored) and ground truth."""
    # This function remains largely the same, showing Z-scored input
    print(f"Saving {num_examples} prediction examples...")
    examples_dir = os.path.join(output_folder, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    try:
        for images, masks in dataset.take(1): # Take one batch
            print(f"Generating predictions for {min(num_examples, images.shape[0])} examples...")
            predictions = model.predict(images) # Model handles its internal preprocessing
            binary_predictions = tf.cast(predictions >= threshold, tf.float32).numpy()
            predictions = predictions.numpy()
            images_numpy = images.numpy() # Z-scored images from dataset
            masks = masks.numpy()

            print("Plotting and saving examples...")
            for j in range(min(num_examples, images_numpy.shape[0])):
                plt.figure(figsize=(16, 4))
                img_zscored = images_numpy[j]
                # Scale Z-scored data approx to [0,1] for display
                p_low, p_high = np.percentile(img_zscored.flatten(), [2, 98])
                img_display = np.clip((img_zscored - p_low) / (p_high - p_low + 1e-8), 0, 1)
                if np.isnan(img_display).any() or np.isinf(img_display).any(): img_display = np.nan_to_num(img_display)

                # Plot 1: Input (Z-scored, Scaled)
                plt.subplot(1, 4, 1)
                plt.title("Input (Z-scored, Scaled)")
                plt.imshow(img_display)
                plt.axis("off")

                # Plot 2: Ground Truth Mask
                plt.subplot(1, 4, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                gt_mask_display = masks[j].squeeze()
                if np.isnan(gt_mask_display).any() or np.isinf(gt_mask_display).any(): gt_mask_display = np.nan_to_num(gt_mask_display)
                plt.imshow(gt_mask_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")

                # Plot 3: Prediction Probabilities
                plt.subplot(1, 4, 3)
                plt.title("Prediction Probabilities")
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                pred_prob_display = predictions[j].squeeze()
                if np.isnan(pred_prob_display).any() or np.isinf(pred_prob_display).any(): pred_prob_display = np.nan_to_num(pred_prob_display)
                prob_map = plt.imshow(pred_prob_display, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")
                # plt.colorbar(prob_map, fraction=0.046, pad=0.04) # Optional

                # Plot 4: Binary Prediction
                plt.subplot(1, 4, 4)
                plt.title(f"Binary Prediction (t={threshold:.2f})")
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                binary_pred_display = binary_predictions[j].squeeze()
                if np.isnan(binary_pred_display).any() or np.isinf(binary_pred_display).any(): binary_pred_display = np.nan_to_num(binary_pred_display)
                plt.imshow(binary_pred_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")

                plt.tight_layout(pad=0.5)
                example_save_path = os.path.join(examples_dir, f"prediction_example_{j+1}.png")
                plt.savefig(example_save_path, dpi=150, bbox_inches='tight')
                plt.close()

            print(f"Saved prediction examples to {examples_dir}")
            break # Only one batch

    except Exception as e:
        print(f"Error saving prediction examples: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()

# ## Completion Notification

def create_completion_notification(output_folder=OUTPUT_DIR, completion_file=COMPLETION_FILE, start_time=None):
    """Creates a text file summarizing the training run and results."""
    print("\n--- Creating Completion Notification ---")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perf_file_path = os.path.join(output_folder, "performances.txt")

    duration_str = "Unknown (start time not recorded)"
    if start_time is not None:
        duration_seconds = time.time() - start_time
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"

    try:
        with open(completion_file, "w") as f:
            f.write(f"AS-Net ({VARIANT_SUFFIX}) Training Completed at: {timestamp}\n\n")
            f.write("Training Configuration:\n")
            f.write(f"- Model: AS-Net with {EFFICIENTNET_VARIANT} encoder\n") # Use variant name
            f.write(f"- Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}\n") # Use correct size
            f.write(f"- Input Channels: {INPUT_CHANNELS}\n")
            f.write(f"- Batch size (per replica): {BATCH_SIZE}\n")
            f.write(f"- Global Batch size: {BATCH_SIZE * strategy.num_replicas_in_sync}\n")
            f.write(f"- Epochs planned: {NUM_EPOCHS}\n")
            f.write(f"- Initial Learning rate: {LEARNING_RATE}\n")
            f.write(f"- Mixed Precision Policy: {mixed_precision.global_policy().name}\n")
            f.write(f"- Loss Config: {COMBINED_LOSS_WEIGHTS}\n")
            f.write(f"- Total Duration: {duration_str}\n\n")

            f.write("Checkpoint and output locations:\n")
            f.write(f"- Checkpoint directory: {CHECKPOINT_DIR}\n") # Use variant path
            f.write(f"- Best model weights: {CHECKPOINT_BEST_PATH}\n") # Use variant path
            f.write(f"- Output directory: {output_folder}\n") # Use variant path

            f.write("\n--- Final Performance Metrics ---\n")
            if os.path.exists(perf_file_path):
                try:
                    with open(perf_file_path, "r") as perf_file:
                        f.write(perf_file.read())
                except Exception as read_err:
                    f.write(f"Note: Error reading performance file ({perf_file_path}): {read_err}\n")
            else:
                f.write(f"Note: Performance file not found ({perf_file_path}). Evaluation failed or not run.\n")

        print(f"Completion notification saved to: {completion_file}")

    except Exception as e:
        print(f"Error creating completion notification file: {e}")
# -

# ## Execute Training and Evaluation

# +
# Record start time
script_start_time = time.time()

# Step 1: Train the model (or skip if complete)
model = None
history = None

if os.path.exists(COMPLETION_FILE):
     print(f"Completion file '{COMPLETION_FILE}' found. Skipping training.")
else:
     print(f"Completion file '{COMPLETION_FILE}' not found. Starting training process...")
     # Pass EFFICIENTNET_VARIANT to the training function
     history, model = train_model_distributed(
         efficientnet_variant=EFFICIENTNET_VARIANT,
         img_height=IMG_HEIGHT, # Pass correct dimensions
         img_width=IMG_WIDTH
     )

# Step 2: Evaluate the model
# Pass EFFICIENTNET_VARIANT and dimensions to evaluation
evaluation_results = evaluate_model(
    model=model, # Pass trained model if available, otherwise it loads from checkpoint
    efficientnet_variant=EFFICIENTNET_VARIANT,
    img_height=IMG_HEIGHT, # Pass correct dimensions
    img_width=IMG_WIDTH
)

# Step 3: Create completion notification
create_completion_notification(start_time=script_start_time)

# Final cleanup
print("\n--- Final Script Cleanup ---")
# Clear potentially large objects
if 'model' in locals() and model is not None: del model
if 'train_dataset' in locals(): del train_dataset
if 'val_dataset' in locals(): del val_dataset
if 'evaluation_results' in locals(): del evaluation_results
if 'history' in locals(): del history
gc.collect()
backend.clear_session()
print(f"Script execution completed successfully for {VARIANT_SUFFIX}!")
# -