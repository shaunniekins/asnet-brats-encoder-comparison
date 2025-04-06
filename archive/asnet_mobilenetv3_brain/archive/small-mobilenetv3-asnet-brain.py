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
# from keras.applications import VGG16 # REMOVED VGG16
from keras.applications import MobileNetV3Large, MobileNetV3Small # ADDED MobileNetV3
from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D, # Keep if needed by custom modules, maybe not by main arch
    Activation,
    UpSampling2D,
    concatenate,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    Add, # Import Add layer
)
# ADDED MobileNetV3 preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input
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
# MOBILENET_VARIANT = 'Large' # Choose 'Large' or 'Small'
MOBILENET_VARIANT = 'Small' # Uncomment to use Small variant

# --- Input Dimensions ---
# Standard MobileNetV3 input size
IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_CHANNELS = 3 # For MobileNetV3 input

# --- Data Loading Constants ---
# Select modalities (Indices based on BraTS documentation: 0:T1, 1:T1ce, 2:T2, 3:FLAIR)
MODALITY_INDICES = [1, 3]  # Use T1ce and FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES) # Should be 2
# Define which loaded modality corresponds to which RGB channel for MobileNetV3 input
# Example: [T1ce, FLAIR, T1ce] -> Index 0, Index 1, Index 0 from the loaded modalities
RGB_MAPPING_INDICES = [0, 1, 0] # Map T1ce to R and B, FLAIR to G

# --- Training Constants ---
# Adjust batch size based on GPU memory with 224x224 input
BATCH_SIZE = 4      # Start low, may increase if memory allows
LEARNING_RATE = 1e-4 # Initial learning rate (Keep based on VGG16 success)
NUM_EPOCHS = 30     # Number of training epochs (or until early stopping)
BUFFER_SIZE = 300   # Shuffle buffer size (adjust based on memory)
THRESHOLD = 0.5     # Segmentation threshold for binary metrics and visualization

# --- Loss Weights ---
# Use weights from successful VGG16 run
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5, 'dice_weight': 0.5, 'class_weight': 100.0} # Tunable loss weights

# --- Paths ---
# Parameterize paths based on the MobileNetV3 variant
VARIANT_SUFFIX = f"mobilenetv3{MOBILENET_VARIANT.lower()}"
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

        # Optional: Limit GPU memory (uncomment if needed)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)] # Example: 4GB limit
        # )
        # print("Limited GPU memory for logical device.")

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

# ## Define AS-Net Model Architecture with MobileNetV3 Encoder

# +
# Helper function to find layer names (useful during development)
def find_layer_names(model, substring):
    names = []
    for layer in model.layers:
        if substring in layer.name:
            names.append(layer.name)
    return names

def AS_Net(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), variant=MOBILENET_VARIANT):
    """Defines the AS-Net model with a MobileNetV3 encoder."""
    # Create an explicit input tensor with a name
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Preprocess input for MobileNetV3 (-1 to 1 scaling)
    preprocessed_inputs = mobilenet_preprocess_input(inputs)

    # Load MobileNetV3 backbone pre-trained on ImageNet
    if variant == 'Large':
        print("Loading MobileNetV3-Large encoder...")
        base_model = MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs  # Pass preprocessed_inputs directly to the base model
        )
        # Define skip connection layers for MobileNetV3-Large
        skip_layer_names = [
            're_lu', # ~H/2 (112x112) output
            'expanded_conv_2_add', # ~H/4 (56x56)
            'expanded_conv_5_add', # ~H/8 (28x28)
            'expanded_conv_14_add', # ~H/16 (14x14) - or actually H/32 (7x7)
            'activation_19'  # ~H/32 (7x7)
        ]
        fine_tune_start_layer_name = 'expanded_conv_6_expand'
    elif variant == 'Small':
        print("Loading MobileNetV3-Small encoder...")
        base_model = MobileNetV3Small(
            weights="imagenet",
            include_top=False,
            input_shape=input_size,
            input_tensor=preprocessed_inputs  # Pass preprocessed_inputs directly
        )
        # Updated skip connection layers for MobileNetV3-Small with correct layer names
        skip_layer_names = [
            're_lu', # ~H/2 (112x112) - first ReLU activation
            'expanded_conv_1_project_bn', # ~H/4 (56x56)
            'expanded_conv_3_project_bn', # ~H/8 (28x28)
            'expanded_conv_7_add', # ~H/16 (14x14) - Changed from expanded_conv_8_add
            'activation_17' # ~H/32 (7x7) - final activation
        ]
        fine_tune_start_layer_name = 'expanded_conv_4_expand'
    else:
        raise ValueError("Invalid MOBILENET_VARIANT. Choose 'Large' or 'Small'.")

    # Debug prints
    print("--- Available layers in MobileNetV3 model ---")
    print("First 5 layers:", [layer.name for layer in base_model.layers[:5]])
    print("Last 5 layers:", [layer.name for layer in base_model.layers[-5:]])
    print(f"Total layers: {len(base_model.layers)}")
    print("Attempting to use skip connections from layers:", skip_layer_names)
    
    # Set trainability based on fine-tuning strategy 
    # ...existing code...
    
    # Extract feature maps from MobileNetV3 encoder stages
    encoder_outputs = []
    for name in skip_layer_names:
        try:
            layer_output = base_model.get_layer(name).output
            encoder_outputs.append(layer_output)
        except ValueError as e:
            print(f"ERROR: Could not find layer {name}. Error: {e}")
            # Print available layer names to help debugging
            print(f"Available layers containing '{name.split('_')[0]}': {[l.name for l in base_model.layers if name.split('_')[0] in l.name]}")
            raise
    
    # Debug print shapes to understand feature map dimensions
    print("\n--- Debugging Feature Map Shapes ---")
    for i, name in enumerate(skip_layer_names):
        output_shape = encoder_outputs[i].shape
        print(f"Layer {name}: Shape {output_shape}")
    
    # Adapt the decoder based on actual feature map shapes
    if variant == 'Large':
        # Unpack encoder outputs
        output1, output2, output3, output4, bottleneck = encoder_outputs
        print(f"Feature maps - H/2:{output1.shape}, H/4:{output2.shape}, H/8:{output3.shape}, H/16:{output4.shape}, H/32:{bottleneck.shape}")
        
        # Check if shapes match our expectations
        if output4.shape[1] == 7:  # If output4 is actually 7x7 (H/32) instead of 14x14 (H/16)
            print("WARNING: Feature map shapes don't match expectations. Both 'H/16' and 'H/32' are actually at 7x7 resolution.")
            print("Adjusting decoder by combining both H/32 feature maps...")
            
            # Combine both H/32 feature maps
            h32_features = concatenate([output4, bottleneck], axis=-1, name='h32_combine')
            print(f"Combined H/32 features shape: {h32_features.shape}")
            
            # Upscale to H/16
            up16 = UpSampling2D((2, 2), interpolation="bilinear", name='up_to_h16')(h32_features)
            print(f"Upsampled to H/16 shape: {up16.shape}")
            
            # SAM and CAM at H/16 level
            filters_h16 = up16.shape[-1]
            SAM4 = SAM(filters=filters_h16, name='sam4')(up16)
            CAM4 = CAM(filters=filters_h16, name='cam4')(up16)
            
            # Rest of the decoder path follows the same pattern as before
            # H/16 -> H/8
            up_sam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
            up_cam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
            merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
            merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
            filters3 = merge31.shape[-1]
            print(f"Decoder stage 2 (H/8): Input filters={filters3}")
            SAM3 = SAM(filters=filters3, name='sam3')(merge31)
            CAM3 = CAM(filters=filters3, name='cam3')(merge32)
            
            # H/8 -> H/4
            up_sam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
            up_cam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
            merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
            merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
            filters2 = merge21.shape[-1]
            print(f"Decoder stage 3 (H/4): Input filters={filters2}")
            SAM2 = SAM(filters=filters2, name='sam2')(merge21)
            CAM2 = CAM(filters=filters2, name='cam2')(merge22)
            
            # H/4 -> H/2
            up_sam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
            up_cam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
            merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
            merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
            filters1 = merge11.shape[-1]
            print(f"Decoder stage 4 (H/2): Input filters={filters1}")
            SAM1 = SAM(filters=filters1, name='sam1')(merge11)
            CAM1 = CAM(filters=filters1, name='cam1')(merge12)
            
            # H/2 -> H/1 (full resolution)
            final_up_sam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
            final_up_cam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
            
        else:
            # Original flow if feature maps match expectations
            print("Feature map shapes as expected. Using original decoder architecture.")
            current_features = bottleneck
            
            # Decoder Stage 1: H/32 -> H/16
            up4 = UpSampling2D((2, 2), interpolation="bilinear", name='up4')(current_features)
            merge4 = concatenate([output4, up4], axis=-1, name='merge4')
            filters4 = merge4.shape[-1]
            print(f"Decoder stage 1 (H/16): Input filters={filters4}")
            SAM4 = SAM(filters=filters4, name='sam4')(merge4)
            CAM4 = CAM(filters=filters4, name='cam4')(merge4)
            
            # Continue with the rest of the decoder stages
            # ... (rest of decoder stages follow the same pattern)
            # ... (code similar to the adjusted path above)
            
            # Prepare final outputs
            final_up_sam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
            final_up_cam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
        
        # Final synergy module
        synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam]) 
    
    # Small variant handling (similar pattern)
    elif variant == 'Small':
        # Debug print shapes to understand feature map dimensions
        for i, name in enumerate(skip_layer_names):
            output_shape = encoder_outputs[i].shape
            print(f"Layer {name}: Shape {output_shape}")
            
        # Unpack encoder outputs
        output1, output2, output3, output4, bottleneck = encoder_outputs
        print(f"Feature maps - H/2:{output1.shape}, H/4:{output2.shape}, H/8:{output3.shape}, H/16:{output4.shape}, H/32:{bottleneck.shape}")
        
        # Check if shapes match our expectations for Small variant
        if output4.shape[1] == 7:  # If output4 is actually 7x7 (H/32) instead of 14x14 (H/16)
            print("WARNING: Feature map shapes don't match expectations. Both 'H/16' and 'H/32' are actually at 7x7 resolution.")
            print("Adjusting decoder by combining both H/32 feature maps...")
            
            # Combine both H/32 feature maps
            h32_features = concatenate([output4, bottleneck], axis=-1, name='h32_combine')
            print(f"Combined H/32 features shape: {h32_features.shape}")
            
            # Upscale to H/16
            up16 = UpSampling2D((2, 2), interpolation="bilinear", name='up_to_h16')(h32_features)
            print(f"Upsampled to H/16 shape: {up16.shape}")
            
            # Apply SAM and CAM at H/16 level (reusing the pattern from Large variant)
            filters_h16 = up16.shape[-1]
            SAM4 = SAM(filters=filters_h16, name='sam4')(up16)
            CAM4 = CAM(filters=filters_h16, name='cam4')(up16)
            
            # Rest of decoder path (H/16 -> H/8 -> H/4 -> H/2 -> final)
            # H/16 -> H/8
            up_sam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
            up_cam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
            merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
            merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
            filters3 = merge31.shape[-1]
            print(f"Decoder stage 2 (H/8): Input filters={filters3}")
            SAM3 = SAM(filters=filters3, name='sam3')(merge31)
            CAM3 = CAM(filters=filters3, name='cam3')(merge32)
            
            # Continue with the same pattern as the Large variant for remaining upsampling stages
            # H/8 -> H/4
            up_sam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
            up_cam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
            merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
            merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
            filters2 = merge21.shape[-1]
            print(f"Decoder stage 3 (H/4): Input filters={filters2}")
            SAM2 = SAM(filters=filters2, name='sam2')(merge21)
            CAM2 = CAM(filters=filters2, name='cam2')(merge22)
            
            # H/4 -> H/2
            up_sam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
            up_cam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
            merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
            merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
            filters1 = merge11.shape[-1]
            print(f"Decoder stage 4 (H/2): Input filters={filters1}")
            SAM1 = SAM(filters=filters1, name='sam1')(merge11)
            CAM1 = CAM(filters=filters1, name='cam1')(merge12)
            
            # H/2 -> H/1 (full resolution)
            final_up_sam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
            final_up_cam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
            
        else:
            # Original flow if feature maps match expectations
            print("Feature map shapes for Small variant - initializing standard decoder architecture.")
            
            # For MobileNetV3-Small based on the feature map logs:
            # - output1 is H/4 (56x56)
            # - output2 is H/8 (28x28)
            # - output3 is H/16 (14x14)
            # - output4 is also H/16 but different channels (14x14)
            # - bottleneck is H/32 (7x7)
            
            # Decoder Stage 1: H/32 -> H/16
            up5 = UpSampling2D((2, 2), interpolation="bilinear", name='up5')(bottleneck)
            # Combine with both H/16 feature maps since we have two at this level
            merge4 = concatenate([output3, output4, up5], axis=-1, name='merge4')
            filters4 = merge4.shape[-1]
            print(f"Decoder stage 1 (H/16): Input filters={filters4}")
            SAM4 = SAM(filters=filters4, name='sam4')(merge4)
            CAM4 = CAM(filters=filters4, name='cam4')(merge4)
            
            # Decoder Stage 2: H/16 -> H/8
            up_sam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
            up_cam4 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
            merge21 = concatenate([output2, up_sam4], axis=-1, name='merge21')
            merge22 = concatenate([output2, up_cam4], axis=-1, name='merge22')
            filters2 = merge21.shape[-1]
            print(f"Decoder stage 2 (H/8): Input filters={filters2}")
            SAM2 = SAM(filters=filters2, name='sam2')(merge21)
            CAM2 = CAM(filters=filters2, name='cam2')(merge22)
            
            # Decoder Stage 3: H/8 -> H/4
            up_sam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
            up_cam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
            merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
            merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
            filters1 = merge11.shape[-1]
            print(f"Decoder stage 3 (H/4): Input filters={filters1}")
            SAM1 = SAM(filters=filters1, name='sam1')(merge11)
            CAM1 = CAM(filters=filters1, name='cam1')(merge12)
            
            # Decoder Stage 4: H/4 -> H/1 (need to upscale by 4x)
            # First go from H/4 -> H/2
            up_sam1 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam1')(SAM1)
            up_cam1 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam1')(CAM1)
            
            # Then from H/2 -> H/1 (final resolution)
            final_up_sam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_sam')(up_sam1)
            final_up_cam = UpSampling2D((2, 2), interpolation="bilinear", name='final_up_cam')(up_cam1)
        
        # Final synergy module for Small variant (same as Large)
        synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam])

    # Final 1x1 convolution for segmentation map (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid", name='final_output', dtype='float32')(synergy_output)

    # Create the model with explicit input and output
    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{variant}')

    # Clean up memory
    gc.collect()
    
    return model


# --- SAM Module ---
# (Keep SAM, CAM, Synergy classes exactly as they were in the VGG16 version)
class SAM(Model):
    """Spatial Attention Module"""
    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        # Output channels for SAM/CAM in AS-Net paper (apply reduction to concatenated input filters)
        self.out_channels = max(16, filters // 4) # Ensure at least 16 channels

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        # Use consistent kernel initialization and padding
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3') # F'(X) path
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # F''(X) path (shortcut)

        # Pooling and Upsampling
        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.upsample1 = UpSampling2D((2, 2), interpolation="bilinear", name='upsample1')
        self.W1 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W1') # Output 1 channel for attention map

        self.pool2 = MaxPooling2D((4, 4), name='pool2')
        self.upsample2 = UpSampling2D((4, 4), interpolation="bilinear", name='upsample2')
        self.W2 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W2') # Output 1 channel for attention map

        self.add_attention = Add(name='add_attention')
        self.multiply = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs))) # Main feature path F'(X)
        out2 = self.conv4(inputs) # Shortcut path F''(X)

        # Parallel attention branches (on shortcut path F''(X))
        pool1 = self.pool1(out2)
        up1 = self.upsample1(pool1)
        # Resize explicitly if shapes might mismatch due to padding/strides
        up1 = tf.image.resize(up1, size=tf.shape(out2)[1:3], method="bilinear")
        att1 = self.W1(up1) # S1 (shape: B, H, W, 1)

        pool2 = self.pool2(out2)
        up2 = self.upsample2(pool2)
        up2 = tf.image.resize(up2, size=tf.shape(out2)[1:3], method="bilinear")
        att2 = self.W2(up2) # S2 (shape: B, H, W, 1)

        # Combine attention weights (S)
        attention_map = self.add_attention([att1, att2]) # Element-wise addition S = S1 + S2 (shape: B, H, W, 1)

        # Apply attention: Multiply main path by attention weights and add shortcut
        # Y = F'(X) * S + F''(X)
        # Attention map (B,H,W,1) is broadcasted across channels of out1 (B,H,W,C_out)
        attended_features = self.multiply([out1, attention_map]) # F'(X) * S
        y = self.add_residual([attended_features, out2]) # Add shortcut connection F''(X)
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
        # Output channels for SAM/CAM in AS-Net paper (apply reduction to concatenated input filters)
        self.out_channels = max(16, filters // 4) # Ensure at least 16 channels
        self.reduction_ratio = reduction_ratio

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3') # F'(X) path
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # F''(X) path (shortcut)

        # Channel attention mechanism
        self.gpool = GlobalAveragePooling2D(name='global_avg_pool', keepdims=True) # Keep dims for easier broadcasting later
        # Dense layers for channel attention weights (use compute dtype)
        reduced_channels = max(1, self.out_channels // self.reduction_ratio) # Ensure at least 1 channel
        self.fc1 = Dense(reduced_channels, activation="relu", use_bias=False, dtype=compute_dtype, name='fc1')
        self.fc2 = Dense(self.out_channels, activation="sigmoid", use_bias=False, dtype=compute_dtype, name='fc2')
        # Reshape is implicitly handled by keepdims=True in gpool and Dense layer shapes

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
        # Channel attention weights are broadcasted across spatial dimensions
        recalibrated_features = self.multiply([out1, channel_attention_weights]) # F'(X) * C
        y = self.add_residual([recalibrated_features, out2]) # Add shortcut connection F''(X)
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
        self.conv = Conv2D(1, 1, padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv') # Output 1 channel
        self.bn = BatchNormalization(name='bn')
        self.add = Add(name='add_weighted') # Use Add layer

    def call(self, inputs):
        sam_features, cam_features = inputs # Expecting a tuple/list of (SAM_output, CAM_output)

        # Cast learnable weights to the compute dtype of the inputs for multiplication
        compute_dtype = sam_features.dtype # Get dtype from input tensor
        alpha_casted = tf.cast(self.alpha, compute_dtype)
        beta_casted = tf.cast(self.beta, compute_dtype)

        # Weighted sum: alpha * SAM_output + beta * CAM_output
        weighted_sum = self.add([alpha_casted * sam_features, beta_casted * cam_features])

        # Apply Conv -> BN
        convolved = self.conv(weighted_sum)
        bn_out = self.bn(convolved)
        # No activation here, final sigmoid is in the main model output layer
        return bn_out # Output has 1 channel (B, H, W, 1)

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
    """Computes the Dice Loss, a common metric for segmentation tasks."""
    def __init__(self, smooth=1e-6, name='dice_loss', **kwargs):
        super(DiceLoss, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Ensure same dtype

        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        # Calculate Dice coefficient and loss
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

        # Clip predictions to avoid log(0) issues
        epsilon_ = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

        # Calculate standard BCE
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Apply weight: Multiply loss for positive examples (y_true == 1) by the weight
        # weight_vector = y_true * self.weight + (1.0 - y_true) * 1.0 # Weight positive class
        # tf.print("Weight:", self.weight)
        # tf.print("y_true shape:", tf.shape(y_true))
        # tf.print("y_pred shape:", tf.shape(y_pred))

        # Calculate weighted BCE using tf.nn.weighted_cross_entropy_with_logits
        # Convert sigmoid output (y_pred) back to logits
        logits = tf.math.log(y_pred / (1.0 - y_pred))

        # TF function expects labels (y_true) and logits
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=tf.cast(self.weight, logits.dtype) # Cast weight to logits dtype
        )
        # tf.print("Weighted BCE Loss shape (per element):", tf.shape(loss))

        # The function returns loss per element, so reduce it (mean over batch)
        # The reduction='sum_over_batch_size' in __init__ handles the batch averaging correctly
        return loss # Return per-element loss, framework handles reduction


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
        # Instantiate WBCE with class weight within the combined loss
        self.wbce = WBCE(weight=class_weight)
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        # Calculate individual losses
        bce_loss_val = self.wbce(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)

        # Combine losses with weights
        # Note: Framework handles the reduction based on 'sum_over_batch_size'
        combined = (self.bce_weight * bce_loss_val) + (self.dice_weight * dice_loss_val)
        return combined

    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "class_weight": self.wbce.weight.numpy() # Get class weight from WBCE instance
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
        # Use state variables to accumulate counts over batches
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros', dtype=tf.float32)
        self.union_sum = self.add_weight(name='union_sum', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32) # Use float32 for internal calculations
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply threshold to predictions
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_binary, [-1])

        # Calculate intersection and sum for the current batch
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        pred_sum = tf.reduce_sum(y_pred_f)
        true_sum = tf.reduce_sum(y_true_f)

        # Update state variables
        self.intersection_sum.assign_add(intersection)
        self.union_sum.assign_add(true_sum + pred_sum)

    def result(self):
        # Calculate Dice coefficient from accumulated sums
        dice = (2.0 * self.intersection_sum + self.smooth) / (self.union_sum + self.smooth)
        # Return result potentially casted to the metric's dtype
        return tf.cast(dice, self._dtype) if self._dtype else dice


    def reset_state(self):
        # Reset state variables at the start of each epoch
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

        # Union = Sum(A) + Sum(B) - Intersection(A,B)
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

# ## Data Preparation for BraTS (with MobileNetV3 Preprocessing)

# +
def prepare_brats_data_gpu(
    metadata_file=METADATA_FILE,
    h5_dir=H5_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Use updated target size
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    modality_indices=MODALITY_INDICES, # Indices to load
    rgb_mapping_indices=RGB_MAPPING_INDICES, # How loaded modalities map to RGB
    num_modalities_loaded=NUM_MODALITIES_LOADED,
    input_channels=INPUT_CHANNELS, # Should be 3 for MobileNetV3
    validation_split=0.2,
    random_seed=42,
):
    """
    Prepares the BraTS 2020 dataset from H5 slices using tf.data.
    Includes MobileNetV3 preprocessing.
    """
    print("--- Setting up Data Pipeline ---")
    print(f"Loading metadata from: {metadata_file}")
    print(f"Loading H5 data from: {h5_dir}")
    print(f"Target image size: {target_size}")
    print(f"Modalities loaded (indices): {modality_indices}")
    print(f"Modalities mapped to RGB (indices): {rgb_mapping_indices}")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"H5 data directory not found at {h5_dir}")

    # Read metadata and filter for existing files
    df = pd.read_csv(metadata_file)
    df['full_path'] = df['slice_path'].apply(lambda x: os.path.join(h5_dir, os.path.basename(x)))
    df = df[df['full_path'].apply(os.path.exists)].copy() # Filter missing files

    if df.empty:
        raise ValueError(f"No valid H5 files found based on metadata in {h5_dir}. Check paths.")
    print(f"Found {len(df)} existing H5 files referenced in metadata.")

    # Shuffle and split data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - validation_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_files = train_df["full_path"].tolist()
    val_files = val_df["full_path"].tolist()

    # Function to parse H5 file (remains the same as VGG16 version)
    def parse_h5_file(file_path):
        def _parse_h5(path_tensor):
            path = path_tensor.numpy().decode("utf-8")
            try:
                with h5py.File(path, "r") as hf:
                    image_data = hf["image"][()] # Shape (H, W, 4) - float64
                    mask_data = hf["mask"][()]   # Shape (H, W, 3) - uint8
                    # Select modalities
                    selected_modalities = image_data[:, :, modality_indices].astype(np.float32)
                    # Create binary mask
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
        original_h, original_w = 240, 240
        image.set_shape([original_h, original_w, num_modalities_loaded])
        mask.set_shape([original_h, original_w])
        return image, mask

    # Function to preprocess image and mask tensors, INCLUDING MobileNetV3 preprocessing
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
        image_rgb = tf.stack(rgb_channels, axis=-1) # Shape: (H, W, 3)

        # --- Resizing ---
        image_resized = tf.image.resize(image_rgb, target_size, method='bilinear')
        mask_expanded = tf.expand_dims(mask, axis=-1)
        mask_resized = tf.image.resize(mask_expanded, target_size, method='nearest')
        mask_final = tf.squeeze(mask_resized, axis=-1)
        mask_final = tf.cast(mask_final > 0.5, tf.float32) # Ensure binary
        mask_final = tf.expand_dims(mask_final, axis=-1) # Add channel dim (H, W, 1)

        # --- MobileNetV3 Specific Preprocessing ---
        # Apply the scaling expected by MobileNetV3 (-1 to 1)
        # IMPORTANT: This step was moved INSIDE the model definition for TF>2.7 compatibility with Keras apps
        # image_preprocessed = mobilenet_preprocess_input(image_resized)

        # For data pipeline, just ensure image is float32 after resize
        image_final = tf.cast(image_resized, tf.float32)

        # --- Set Final Shapes ---
        image_final.set_shape([target_size[0], target_size[1], input_channels])
        mask_final.set_shape([target_size[0], target_size[1], 1])

        # tf.print("Preprocessed shapes - Image:", tf.shape(image_final), "Mask:", tf.shape(mask_final))
        # tf.print("Preprocessed types - Image:", image_final.dtype, "Mask:", mask_final.dtype)
        # tf.print("Image range (before preprocess_input):", tf.reduce_min(image_final), tf.reduce_max(image_final))

        return image_final, mask_final # Return image BEFORE mobilenet preprocessing

    # --- Create tf.data Datasets ---
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_files)

    # --- Apply Transformations ---
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = (
        train_dataset.with_options(options)
        .shuffle(buffer_size) # Shuffle file paths BEFORE loading for better randomness
        .map(parse_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        # Removed shuffle here, shuffle files instead
        .batch(batch_size)    # Batch data AFTER preprocessing
        .prefetch(tf.data.AUTOTUNE) # Prefetch for performance
    )
    val_dataset = (
        val_dataset.with_options(options)
        .map(parse_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Data pipeline created successfully.")
    # Optional: Inspect element spec
    print("Train Dataset Element Spec:", train_dataset.element_spec)
    print("Validation Dataset Element Spec:", val_dataset.element_spec)

    return train_dataset, val_dataset

# Function to visualize samples from the dataset
def visualize_dataset_samples(dataset, num_samples=3, output_dir=OUTPUT_DIR):
    """Visualizes samples from the dataset and saves the plot."""
    print("--- Visualizing Dataset Samples ---")
    try:
        plt.figure(figsize=(15, 5 * num_samples))
        plot_count = 0
        for images, masks in dataset.take(1): # Take one batch
            num_in_batch = images.shape[0]
            print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
            print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}")
            # Note: Image range here is Z-scored data, BEFORE MobileNet preprocess_input
            print(f"Image value range (Z-scored): Min={tf.reduce_min(images):.4f}, Max={tf.reduce_max(images):.4f}")
            print(f"Mask value range: Min={tf.reduce_min(masks):.4f}, Max={tf.reduce_max(masks):.4f}")
            print(f"Unique mask values in batch: {tf.unique(tf.reshape(masks, [-1]))[0]}")

            for i in range(min(num_samples, num_in_batch)):
                plot_count += 1
                img_zscored = images[i].numpy()
                mask = masks[i].numpy().squeeze() # Remove channel dim for plotting

                # --- Display individual channels and combined RGB (Z-scored) ---
                titles = ["Input R (T1ce Z)", "Input G (FLAIR Z)", "Input B (T1ce Z)", "RGB Input (Z-scored)", "Ground Truth Mask"]
                channels_to_plot = [img_zscored[:, :, 0], img_zscored[:, :, 1], img_zscored[:, :, 2], img_zscored, mask]

                for j, item in enumerate(channels_to_plot):
                    ax = plt.subplot(num_samples, len(titles), i * len(titles) + j + 1)
                    ax.set_title(titles[j])

                    if j < 4: # Image data (Z-scored)
                        # Clip Z-score normalized data to roughly [0,1] for display via percentiles
                        p_low, p_high = np.percentile(item.flatten(), [2, 98]) # Use flatten for RGB
                        item_disp = np.clip((item - p_low) / (p_high - p_low + 1e-8), 0, 1)
                        if np.isnan(item_disp).any() or np.isinf(item_disp).any():
                            print(f"Warning: NaN/Inf found in image display for sample {i}, plot {j}")
                            item_disp = np.nan_to_num(item_disp) # Replace NaN/Inf
                        plt.imshow(item_disp, cmap='gray' if item.ndim==2 else None) # Use cmap only for single channel
                    else: # Mask
                        if np.isnan(item).any() or np.isinf(item).any():
                            print(f"Warning: NaN/Inf found in mask display for sample {i}")
                            item = np.nan_to_num(item)
                        plt.imshow(item, cmap='jet', vmin=0, vmax=1) # Use 'jet' or 'viridis' for mask

                    plt.axis("off")

        plt.tight_layout(pad=0.5)
        save_path = os.path.join(output_dir, "dataset_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # Close plot to free memory
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
    def __init__(self, log_frequency=1): # Log every epoch by default
        super().__init__()
        self.log_frequency = log_frequency  # Fixed: assign parameter to attribute
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

            # Monitor Synergy weights
            try:
                synergy_layer = self.model.get_layer('synergy')
                alpha_val = synergy_layer.alpha.numpy()
                beta_val = synergy_layer.beta.numpy()
                print(f"    Synergy weights - alpha: {alpha_val:.4f}, beta: {beta_val:.4f}")
            except Exception as e:
                pass # Layer might not exist

        # Force garbage collection at end of epoch
        gc.collect()

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n--- Training Finished ---")
        print(f"Total training time: {total_time:.2f} seconds")


# Learning Rate Scheduler Function
def lr_step_decay(epoch, lr):
    """Applies step decay to the learning rate."""
    initial_lr = LEARNING_RATE
    drop = 0.5       # Factor to drop learning rate by
    epochs_drop = 10  # Number of epochs before dropping LR
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
    combined_loss_weights=COMBINED_LOSS_WEIGHTS, # Use constant from top
    metrics=['binary_accuracy',
             DiceCoefficient(name='dice_coef'),
             IoU(name='iou'),
             tf.keras.metrics.Precision(thresholds=THRESHOLD, name="precision"),
             tf.keras.metrics.Recall(thresholds=THRESHOLD, name="recall")],
    mobilenet_variant=MOBILENET_VARIANT # Pass variant to model function
):
    """Trains the AS-Net MobileNetV3 model using tf.distribute.Strategy."""

    # Calculate global batch size
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    print(f"--- Starting Training ({VARIANT_SUFFIX}) ---") # Include variant in log
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Loss configuration: {combined_loss_weights}")


    # 1. Create Datasets within Strategy Scope
    print("Preparing datasets...")
    # Pass global_batch_size to the dataset function
    train_dataset, val_dataset = dataset_func(batch_size=global_batch_size)
    print("Datasets prepared.")

    # Optionally visualize dataset samples before training
    visualize_dataset_samples(train_dataset, num_samples=3, output_dir=output_dir)

    # 2. Build and Compile Model within Strategy Scope
    with strategy.scope():
        print("Building model...")
        # Pass variant to the model function
        model = model_func(input_size=(img_height, img_width, input_channels), variant=mobilenet_variant)
        print("Model built.")

        print("Compiling model...")
        # Instantiate loss function within strategy scope
        print("Using Combined WBCE + Dice Loss")
        loss_instance = CombinedLoss(
             bce_weight=combined_loss_weights['bce_weight'],
             dice_weight=combined_loss_weights['dice_weight'],
             class_weight=combined_loss_weights['class_weight']
        )

        # Instantiate optimizer within strategy scope
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Apply loss scaling for mixed precision if enabled
        if mixed_precision.global_policy().name == 'mixed_float16':
             optimizer = mixed_precision.LossScaleOptimizer(optimizer)
             print("Loss scaling applied for mixed precision.")

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss_instance,
            metrics=metrics # Use metrics list defined in function signature
        )
        print("Model compiled.")
        model.summary(line_length=120) # Print model summary

    # Check if a checkpoint exists to resume training
    latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
    initial_epoch = 0
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        try:
            # Load weights within strategy scope if model was created within scope
            with strategy.scope():
                model.load_weights(latest_checkpoint).expect_partial() # Use expect_partial

            # Extract epoch number from checkpoint filename if possible
            try:
                 filename = os.path.basename(latest_checkpoint)
                 # Common patterns: model.10.h5, model_epoch_10.h5, ckpt-10, etc.
                 # Extract numbers, assume the first one before .h5 or end is epoch
                 numbers = [int(s) for s in filename.replace('.weights.h5','').split('_') if s.isdigit()]
                 if not numbers: # Try splitting by '.' or '-'
                      parts = filename.replace('.weights.h5','').replace('-', '.').split('.')
                      numbers = [int(s) for s in parts if s.isdigit()]

                 if numbers:
                     initial_epoch = max(numbers) # Take the largest number as likely epoch
                     print(f"Successfully loaded weights. Starting from epoch {initial_epoch + 1}")
                 else: raise ValueError("No numbers found")

            except Exception as parse_err:
                 print(f"Warning: Could not determine epoch number from checkpoint name '{latest_checkpoint}', starting from epoch 0. Error: {parse_err}")
                 initial_epoch = 0
        except Exception as load_err:
            print(f"Error loading weights from {latest_checkpoint}: {load_err}. Starting training from scratch.")
            initial_epoch = 0
    else:
        print("No checkpoint found, starting training from scratch.")


    # 3. Define Callbacks
    callbacks = [
        # Save best weights based on val_dice_coef (higher is better)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_best_path, # Use variant path
            save_weights_only=True,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Save weights every epoch (for resuming)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, # Use variant path (overwrites each epoch)
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        # Apply learning rate schedule
        tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=0),
        # Custom progress logger
        ConciseProgressCallback(log_frequency=1),
        # TensorBoard (optional)
        # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=1)
    ]

    # 4. Train the Model
    print(f"Starting training loop for max {num_epochs} epochs (current: {initial_epoch})...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        initial_epoch=initial_epoch, # Start from the correct epoch if resuming
        callbacks=callbacks,
        verbose=0 # Use custom callback for progress logging
    )

    # 5. Save Training History
    try:
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = os.path.join(output_dir, 'training_history.csv')
        hist_df.to_csv(hist_csv_file, index=False)
        print(f"Training history saved to {hist_csv_file}")
        # Plot history and save plots
        plot_training_history(history, output_dir)
    except Exception as e:
        print(f"Error saving/plotting training history: {e}")

    # Force cleanup after training
    print(f"Cleaning up resources after training ({VARIANT_SUFFIX})...")
    del train_dataset, val_dataset # Explicitly delete datasets
    gc.collect()
    # Keep model instance if returning it, otherwise clear session
    # backend.clear_session()
    print("Cleaned up datasets.")

    return history, model # Return the final model state

# ## Plot Training History

def plot_training_history(history, output_dir=OUTPUT_DIR):
    """Plots training & validation loss and metrics and saves the plot."""
    print("--- Plotting Training History ---")
    try:
        history_dict = history.history
        if not history_dict:
             print("History object is empty. Skipping plotting.")
             return

        epochs = range(1, len(history_dict['loss']) + 1)

        # Determine available metrics
        metrics_to_plot = {'loss': 'Loss'}
        for key in history_dict.keys():
            if key.startswith('val_'): continue
            if key == 'loss': continue
            # Add known metrics
            if key == 'dice_coef': metrics_to_plot['dice_coef'] = 'Dice Coefficient'
            elif key == 'iou': metrics_to_plot['iou'] = 'IoU (Jaccard)'
            elif key == 'binary_accuracy': metrics_to_plot['binary_accuracy'] = 'Binary Accuracy'
            elif key == 'precision': metrics_to_plot['precision'] = 'Precision'
            elif key == 'recall': metrics_to_plot['recall'] = 'Recall'
            # Add others if needed, using key as title
            elif f'val_{key}' in history_dict: # Check if validation counterpart exists
                 metrics_to_plot[key] = key.replace('_', ' ').title()


        num_plots = len(metrics_to_plot)
        if num_plots <= 1: # Only loss found
            print("Warning: Only 'loss' metric found in history. Plotting loss only.")
            if num_plots == 0: return # Should not happen if loss exists

        plt.figure(figsize=(max(12, 6 * num_plots), 5)) # Adjust figure size

        plot_index = 1
        for metric, title in metrics_to_plot.items():
            plt.subplot(1, num_plots, plot_index)
            val_metric = f'val_{metric}'

            # Plot training metric
            if metric in history_dict:
                plt.plot(epochs, history_dict[metric], 'bo-', label=f'Training {title}')
            else:
                 print(f"Warning: Training metric '{metric}' not found.")

            # Plot validation metric
            if val_metric in history_dict:
                plt.plot(epochs, history_dict[val_metric], 'ro-', label=f'Validation {title}')
            else:
                 # Don't plot legend entry if validation metric is missing
                 if metric in history_dict: plt.legend() # Show legend only for training if val is missing
                 print(f"Warning: Validation metric '{val_metric}' not found.")

            plt.title(f'{title}') # Simplified title
            plt.xlabel('Epoch')
            # plt.ylabel(title) # Title is sufficient
            if metric != 'loss' and val_metric in history_dict: # Only add legend if both train/val exist
                 plt.legend()
            if metric != 'loss': plt.ylim([0, 1.05]) # Limit y-axis for common metrics [0, 1]
            plt.grid(True)
            plot_index += 1

        plt.suptitle(f'AS-Net {MOBILENET_VARIANT} Training History', fontsize=14) # Add main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        save_path = os.path.join(output_dir, "training_history_plots.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # Close plot to free memory
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
    checkpoint_path = CHECKPOINT_PATH, # Last epoch checkpoint path
    output_folder=OUTPUT_DIR, # Use variant path
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    input_channels=INPUT_CHANNELS,
    batch_size=BATCH_SIZE, # Use evaluation batch size
    dataset_func=prepare_brats_data_gpu,
    threshold=THRESHOLD,
    num_examples_to_save=5,
    loss_config=COMBINED_LOSS_WEIGHTS, # Pass loss config
    mobilenet_variant=MOBILENET_VARIANT # Pass variant
):
    """Evaluates the trained AS-Net MobileNetV3 model on the validation set."""
    print(f"\n--- Starting Model Evaluation ({VARIANT_SUFFIX}) ---")
    evaluation_results = None # Initialize results

    try:
        # 1. Load Validation Data
        # Use global batch size for evaluation dataset as well
        global_eval_batch_size = batch_size * strategy.num_replicas_in_sync
        print(f"Loading validation data with batch size: {global_eval_batch_size}...")
        _, val_dataset = dataset_func(batch_size=global_eval_batch_size, validation_split=0.2) # Ensure consistent split
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
                 # Try loading the last epoch checkpoint instead
                 last_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
                 if last_checkpoint and os.path.exists(last_checkpoint + ".index"): # Check index file exists
                      print(f"Attempting to load last epoch checkpoint: {last_checkpoint}")
                      checkpoint_to_load = last_checkpoint
                 else:
                     print(f"Error: No suitable checkpoint found in {os.path.dirname(checkpoint_path)}. Cannot evaluate.")
                     return None # Cannot proceed without a model

             # Build model architecture again before loading weights
             print("Rebuilding model architecture for evaluation...")
             with strategy.scope(): # Load model within strategy scope
                 # Pass variant to the model function
                 model_eval = AS_Net(input_size=(img_height, img_width, input_channels), variant=mobilenet_variant)

                 # Compile is necessary to run evaluate, use the same loss/metrics as training
                 print("Compiling evaluation model...")
                 loss_instance = CombinedLoss(**loss_config) # Use **kwargs for clarity
                 optimizer = tf.keras.optimizers.Adam() # Optimizer state isn't used for eval but needed for compile
                 if mixed_precision.global_policy().name == 'mixed_float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                 model_eval.compile(
                      optimizer=optimizer,
                      loss=loss_instance,
                      metrics=['binary_accuracy',
                               DiceCoefficient(name='dice_coef'), # Ensure metrics are instantiated
                               IoU(name='iou'),
                               tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
                               tf.keras.metrics.Recall(thresholds=threshold, name="recall")]
                 )
                 print("Evaluation model compiled.")
                 print(f"Loading weights from {checkpoint_to_load}...")
                 load_status = model_eval.load_weights(checkpoint_to_load)
                 load_status.expect_partial() # Allow loading even if optimizer state is missing/different
                 # load_status.assert_consumed() # Uncomment to check all weights were loaded
                 print(f"Successfully loaded weights into new model instance from {checkpoint_to_load}")
        else:
            print("Using provided trained model instance for evaluation.")
            model_eval = model # Use the model passed from training
            # Ensure the provided model is compiled if it wasn't already (e.g., if loaded without compiling)
            if not model_eval.optimizer:
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


        # 3. Evaluate using model.evaluate()
        print("Evaluating model on validation set...")
        evaluation_results = model_eval.evaluate(val_dataset, verbose=1, return_dict=True)
        print("\nKeras Evaluation Results:")
        for name, value in evaluation_results.items():
            print(f"- {name}: {value:.4f}")

        # 4. Calculate F1 Score (using precision and recall from Keras metrics)
        precision_val = evaluation_results.get('precision', 0.0)
        recall_val = evaluation_results.get('recall', 0.0)
        f1_val = 0.0
        if (precision_val + recall_val) > 1e-7: # Avoid division by zero
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        evaluation_results['f1_score'] = f1_val # Add F1 to the results dictionary
        print(f"- f1_score: {f1_val:.4f} (calculated from Precision/Recall)")

        # 5. Save Performance Metrics to File
        try:
            perf_file_path = os.path.join(output_folder, "performances.txt")
            with open(perf_file_path, "w") as file_perf:
                file_perf.write(f"Evaluation Metrics ({VARIANT_SUFFIX}):\n")
                file_perf.write("------------------------------------\n")
                for name, value in evaluation_results.items():
                     file_perf.write(f"- {name.replace('_', ' ').title()}: {value:.4f}\n") # Nicer formatting
            print(f"Evaluation results saved to {perf_file_path}")
        except Exception as e:
            print(f"Error saving performance metrics to file: {e}")

        # 6. Generate and Save Prediction Examples
        print("\nGenerating prediction examples...")
        # Ensure the model used for prediction is the evaluated one
        save_prediction_examples(model_eval, val_dataset, output_folder, num_examples=num_examples_to_save, threshold=threshold)

        print(f"--- Evaluation Finished ({VARIANT_SUFFIX}) ---")
        return evaluation_results # Return the metrics dictionary

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None if evaluation failed
    finally:
        # Final cleanup
        print("Cleaning up resources after evaluation...")
        if 'val_dataset' in locals(): del val_dataset # Delete dataset
        if 'model_eval' in locals() and model_eval is not model: del model_eval # Delete model if loaded here
        gc.collect()
        # Avoid clearing session if model was passed in from training and might be used later
        # backend.clear_session()
        print("Cleaned up evaluation resources.")

# ## Save Prediction Examples

def save_prediction_examples(model, dataset, output_folder, num_examples=5, threshold=THRESHOLD):
    """Saves example predictions with inputs and ground truth."""
    print(f"Saving {num_examples} prediction examples...")
    examples_dir = os.path.join(output_folder, "examples")
    os.makedirs(examples_dir, exist_ok=True) # Ensure directory exists

    try:
        # Take one batch from the dataset
        for images, masks in dataset.take(1):
            print(f"Generating predictions for {min(num_examples, images.shape[0])} examples...")
            # Predict using the model
            predictions = model.predict(images) # Images are already preprocessed by the model
            # Apply threshold to get binary predictions
            binary_predictions = tf.cast(predictions >= threshold, tf.float32).numpy()
            predictions = predictions.numpy() # Get probability maps as numpy array
            images_numpy = images.numpy() # Z-scored images from dataset
            masks = masks.numpy()

            print("Plotting and saving examples...")
            for j in range(min(num_examples, images_numpy.shape[0])):
                plt.figure(figsize=(16, 4)) # Width adjusted for 4 plots

                # --- Plot 1: Original Input (Z-scored, scaled for display) ---
                plt.subplot(1, 4, 1)
                plt.title("Input (Z-scored, Scaled)")
                # Scale Z-scored data approx to [0,1] for display using percentiles
                img_zscored = images_numpy[j]
                p_low, p_high = np.percentile(img_zscored.flatten(), [2, 98])
                img_display = np.clip((img_zscored - p_low) / (p_high - p_low + 1e-8), 0, 1)
                if np.isnan(img_display).any() or np.isinf(img_display).any():
                     print(f"Warning: NaN/Inf found in input image display for example {j}")
                     img_display = np.nan_to_num(img_display)
                plt.imshow(img_display)
                plt.axis("off")

                # --- Plot 2: Ground Truth Mask ---
                plt.subplot(1, 4, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(img_display, cmap='gray', alpha=0.6) # Background
                gt_mask_display = masks[j].squeeze()
                if np.isnan(gt_mask_display).any() or np.isinf(gt_mask_display).any():
                     print(f"Warning: NaN/Inf found in GT mask display for example {j}")
                     gt_mask_display = np.nan_to_num(gt_mask_display)
                plt.imshow(gt_mask_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")

                # --- Plot 3: Prediction Probability Map ---
                plt.subplot(1, 4, 3)
                plt.title("Prediction Probabilities")
                plt.imshow(img_display, cmap='gray', alpha=0.6) # Background
                pred_prob_display = predictions[j].squeeze()
                if np.isnan(pred_prob_display).any() or np.isinf(pred_prob_display).any():
                     print(f"Warning: NaN/Inf found in probability map display for example {j}")
                     pred_prob_display = np.nan_to_num(pred_prob_display)
                prob_map = plt.imshow(pred_prob_display, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")
                # plt.colorbar(prob_map, fraction=0.046, pad=0.04) # Optional colorbar

                # --- Plot 4: Binary Prediction (Thresholded) ---
                plt.subplot(1, 4, 4)
                plt.title(f"Binary Prediction (t={threshold:.2f})")
                plt.imshow(img_display, cmap='gray', alpha=0.6) # Background
                binary_pred_display = binary_predictions[j].squeeze()
                if np.isnan(binary_pred_display).any() or np.isinf(binary_pred_display).any():
                     print(f"Warning: NaN/Inf found in binary prediction display for example {j}")
                     binary_pred_display = np.nan_to_num(binary_pred_display)
                plt.imshow(binary_pred_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")

                # Save the figure
                plt.tight_layout(pad=0.5)
                example_save_path = os.path.join(examples_dir, f"prediction_example_{j+1}.png")
                plt.savefig(example_save_path, dpi=150, bbox_inches='tight')
                plt.close() # Close the figure to free memory

            print(f"Saved prediction examples to {examples_dir}")
            break # Only process one batch

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

    # Calculate duration if start_time was provided
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
            f.write(f"- Model: AS-Net with MobileNetV3-{MOBILENET_VARIANT} encoder\n")
            f.write(f"- Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}\n")
            f.write(f"- Input Channels: {INPUT_CHANNELS}\n")
            f.write(f"- Batch size (per replica): {BATCH_SIZE}\n")
            f.write(f"- Global Batch size: {BATCH_SIZE * strategy.num_replicas_in_sync}\n")
            f.write(f"- Epochs planned: {NUM_EPOCHS}\n")
            f.write(f"- Initial Learning rate: {LEARNING_RATE}\n")
            f.write(f"- Mixed Precision Policy: {mixed_precision.global_policy().name}\n")
            f.write(f"- Loss Config: {COMBINED_LOSS_WEIGHTS}\n")
            f.write(f"- Total Duration: {duration_str}\n\n")

            f.write("Checkpoint and output locations:\n")
            f.write(f"- Checkpoint directory: {CHECKPOINT_DIR}\n")
            f.write(f"- Best model weights: {CHECKPOINT_BEST_PATH}\n")
            f.write(f"- Output directory: {output_folder}\n")

            # Add final performance metrics if available
            f.write("\n--- Final Performance Metrics ---\n")
            if os.path.exists(perf_file_path):
                try:
                    with open(perf_file_path, "r") as perf_file:
                        f.write(perf_file.read())
                except Exception as read_err:
                    f.write(f"Note: Error reading performance metrics file ({perf_file_path}): {read_err}\n")
            else:
                f.write(f"Note: Performance metrics file not found ({perf_file_path}). Evaluation might have failed or not run yet.\n")

        print(f"Completion notification saved to: {completion_file}")

    except Exception as e:
        print(f"Error creating completion notification file: {e}")

# ## Execute Training and Evaluation

# +
# Record start time for duration calculation
script_start_time = time.time()

# Step 1: Train the model (or load if already trained)
model = None # Initialize model variable
history = None # Initialize history variable

if os.path.exists(COMPLETION_FILE):
     print(f"Completion file '{COMPLETION_FILE}' found. Skipping training.")
     # No need to build the model here; evaluation will build it if needed
else:
     print(f"Completion file '{COMPLETION_FILE}' not found. Starting training process...")
     # Pass MOBILENET_VARIANT to the training function
     history, model = train_model_distributed(mobilenet_variant=MOBILENET_VARIANT) # history and model are returned

# Step 2: Evaluate the model
# Pass the model ONLY if training actually ran and returned a model, otherwise evaluate loads from checkpoint
# Pass MOBILENET_VARIANT to the evaluation function
evaluation_results = evaluate_model(model=model, mobilenet_variant=MOBILENET_VARIANT) # Pass model if it exists

# Step 3: Create completion notification (will include eval results if successful)
create_completion_notification(start_time=script_start_time)

# Final cleanup
print("\n--- Final Script Cleanup ---")
if 'model' in locals() and model is not None: del model
if 'train_dataset' in locals(): del train_dataset
if 'val_dataset' in locals(): del val_dataset
if 'evaluation_results' in locals(): del evaluation_results
if 'history' in locals(): del history
gc.collect()
backend.clear_session()
print(f"Script execution completed successfully for {VARIANT_SUFFIX}!")
# -