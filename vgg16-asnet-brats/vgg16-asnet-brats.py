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
from keras.applications import VGG16
from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
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
# Consider smaller dimensions if memory is tight, but ensure aspect ratio is maintained if possible
IMG_HEIGHT = 192
IMG_WIDTH = 192  # Changed to square for simplicity, adjust if necessary
INPUT_CHANNELS = 3 # For VGG16 input

# --- Data Loading Constants ---
# Select modalities (Indices based on BraTS documentation: 0:T1, 1:T1ce, 2:T2, 3:FLAIR)
MODALITY_INDICES = [1, 3]  # Use T1ce and FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES) # Should be 2
# Define which loaded modality corresponds to which RGB channel for VGG16 input
# Example: [T1ce, FLAIR, T1ce] -> Index 0, Index 1, Index 0 from the loaded modalities
RGB_MAPPING_INDICES = [0, 1, 0] # Map T1ce to R and B, FLAIR to G

# --- Training Constants ---
BATCH_SIZE = 4      # Keep low due to memory constraints
LEARNING_RATE = 1e-4 # Initial learning rate (LOWERED)
NUM_EPOCHS = 30     # Number of training epochs (or until early stopping)
BUFFER_SIZE = 300   # Shuffle buffer size (adjust based on memory)
THRESHOLD = 0.5     # Segmentation threshold for binary metrics and visualization

# --- Loss Weights ---
# INCREASED CLASS WEIGHT SIGNIFICANTLY
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5, 'dice_weight': 0.5, 'class_weight': 100.0} # Tunable loss weights
# ALTERNATIVE: Use Focal Loss + Dice
# USE_FOCAL_LOSS = True # Set to True to use Focal loss instead of WBCE
# COMBINED_LOSS_WEIGHTS = {'focal_weight': 0.5, 'dice_weight': 0.5, 'focal_gamma': 2.0}


# --- Paths ---
# Ensure these paths exist or are created
CHECKPOINT_DIR = "./vgg16-checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/vgg16_as_net_model.weights.h5"
CHECKPOINT_BEST_PATH = f"{CHECKPOINT_DIR}/vgg16_as_net_model_best.weights.h5"
OUTPUT_DIR = "vgg16-output"
COMPLETION_FILE = "vgg16-asnet-finished-training.txt"
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

        strategy = tf.distribute.MirroredStrategy() # Use if multiple GPUs
        print(f"Running on {strategy.num_replicas_in_sync} GPU(s) using MirroredStrategy.")
    except RuntimeError as e:
        print(f"GPU Memory Growth Error: {e}. Trying default strategy.")
        strategy = tf.distribute.get_strategy() # Fallback to default
        print("Running on CPU or single GPU (default strategy).")
else:
    strategy = tf.distribute.get_strategy() # Default strategy (CPU or single GPU)
    print("No GPU detected. Running on CPU.")
print("Global Batch Size (per replica * num replicas):", BATCH_SIZE * strategy.num_replicas_in_sync)

print("\n--- Mixed Precision Configuration ---")
# Use mixed precision to reduce memory usage and potentially speed up training on compatible GPUs
# policy = mixed_precision.Policy('mixed_float16')
policy = mixed_precision.Policy('float32') # Using float32 as requested, change if using mixed precision
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy set to: {policy.name}")
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# +
# Configure JIT compilation
# tf.config.optimizer.set_jit(True) # Can improve performance but might use more memory initially
# print("JIT compilation enabled.")
# -

# ## Define AS-Net Model Architecture

# +
def AS_Net(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)):
    """Defines the AS-Net model with a VGG16 encoder."""
    inputs = Input(input_size, dtype=tf.float32) # Ensure input dtype is float32

    # Load VGG16 backbone pre-trained on ImageNet
    # Use compute_dtype from mixed precision policy for the layers
    compute_dtype = mixed_precision.global_policy().compute_dtype
    VGGnet = VGG16(weights="imagenet", include_top=False, input_tensor=inputs) # Pass input tensor directly

    # --- Fine-tuning ---
    # Unfreeze layers from block4 onwards (Recommended Start)
    for layer in VGGnet.layers:
        layer.trainable = False # Freeze initially
    # Unfreeze layers from block4 onwards (index > 10 for typical VGG16)
    for layer in VGGnet.layers[11:]: # Adjust index based on model.summary() if needed
        layer.trainable = True
    print("Unfroze VGG16 layers from block4 onwards for fine-tuning.")
    # --- End Fine-tuning ---

    # Extract feature maps from VGG16 encoder stages
    # Layer indices might need adjustment based on VGG16 implementation details. Verify with model.summary().
    # Typical VGG16 blocks end at: block1_conv2 (idx 2), block2_conv2 (idx 5), block3_conv3 (idx 9), block4_conv3 (idx 13), block5_conv3 (idx 17)
    output1 = VGGnet.get_layer(index=2).output # block1_conv2, Shape: (H, W, 64)
    output2 = VGGnet.get_layer(index=5).output # block2_conv2, Shape: (H/2, W/2, 128)
    output3 = VGGnet.get_layer(index=9).output # block3_conv3, Shape: (H/4, W/4, 256)
    output4 = VGGnet.get_layer(index=13).output # block4_conv3, Shape: (H/8, W/8, 512)
    output5 = VGGnet.get_layer(index=17).output # block5_conv3, Shape: (H/16, W/16, 512)

    # --- Decoder with SAM, CAM, and Synergy ---
    # Upsample block 5, concatenate with block 4
    up5 = UpSampling2D((2, 2), interpolation="bilinear", name='up5')(output5)
    merge1 = concatenate([output4, up5], axis=-1, name='merge1') # Shape: (H/8, W/8, 512+512=1024)

    # Apply SAM and CAM at the first decoder stage
    # Input filters = 1024. Output filters = 1024 // 4 = 256
    SAM1 = SAM(filters=1024, name='sam1')(merge1)
    CAM1 = CAM(filters=1024, name='cam1')(merge1)

    # Upsample SAM1/CAM1, concatenate with block 3
    up_sam1 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam1')(SAM1)
    up_cam1 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam1')(CAM1)
    merge21 = concatenate([output3, up_sam1], axis=-1, name='merge21') # Shape: (H/4, W/4, 256 + 256 = 512)
    merge22 = concatenate([output3, up_cam1], axis=-1, name='merge22') # Shape: (H/4, W/4, 256 + 256 = 512)

    # Apply SAM and CAM at the second decoder stage
    # Input filters = 512. Output filters = 512 // 4 = 128
    SAM2 = SAM(filters=512, name='sam2')(merge21)
    CAM2 = CAM(filters=512, name='cam2')(merge22)

    # Upsample SAM2/CAM2, concatenate with block 2
    up_sam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    merge31 = concatenate([output2, up_sam2], axis=-1, name='merge31') # Shape: (H/2, W/2, 128 + 128 = 256)
    merge32 = concatenate([output2, up_cam2], axis=-1, name='merge32') # Shape: (H/2, W/2, 128 + 128 = 256)

    # Apply SAM and CAM at the third decoder stage
    # Input filters = 256. Output filters = 256 // 4 = 64
    SAM3 = SAM(filters=256, name='sam3')(merge31)
    CAM3 = CAM(filters=256, name='cam3')(merge32)

    # Upsample SAM3/CAM3, concatenate with block 1
    up_sam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D((2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    merge41 = concatenate([output1, up_sam3], axis=-1, name='merge41') # Shape: (H, W, 64 + 64 = 128)
    merge42 = concatenate([output1, up_cam3], axis=-1, name='merge42') # Shape: (H, W, 64 + 64 = 128)

    # Apply SAM and CAM at the fourth decoder stage
    # Input filters = 128. Output filters = 128 // 4 = 32
    SAM4 = SAM(filters=128, name='sam4')(merge41)
    CAM4 = CAM(filters=128, name='cam4')(merge42)

    # Synergy module to combine final SAM and CAM outputs
    synergy_output = Synergy(name='synergy')((SAM4, CAM4)) # Input shapes: (H, W, 32) + (H, W, 32) -> Synergy Conv(filters=1) -> (H, W, 1)

    # Final 1x1 convolution for segmentation map
    # Use float32 for the final output layer for numerical stability, even in mixed precision
    output = Conv2D(1, 1, padding="same", activation="sigmoid", name='final_output', dtype='float32')(synergy_output)

    # Create the model
    model = Model(inputs=inputs, outputs=output, name='AS_Net_VGG16')

    # Clean up memory after building model (optional)
    gc.collect()

    return model

# --- SAM Module ---
class SAM(Model):
    """Spatial Attention Module"""
    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_channels = filters // 4 # Output channels for SAM/CAM in AS-Net paper

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # Branch for attention weights (shortcut path)

        # Pooling and Upsampling
        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.upsample1 = UpSampling2D((2, 2), interpolation="bilinear", name='upsample1')
        self.W1 = Conv2D(self.out_channels, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W1')

        self.pool2 = MaxPooling2D((4, 4), name='pool2')
        self.upsample2 = UpSampling2D((4, 4), interpolation="bilinear", name='upsample2')
        self.W2 = Conv2D(self.out_channels, 1, activation="sigmoid", kernel_initializer="he_normal", dtype=compute_dtype, name='W2')

        self.activation = Activation('relu', name='relu_act')
        self.multiply = Multiply(name='multiply')
        self.add = Add(name='add') # Use Add layer

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs))) # Main feature path F'(X)
        out2 = self.conv4(inputs) # Shortcut path F''(X)

        # Parallel attention branches (on shortcut path F''(X))
        pooled1 = self.pool1(out2)
        upsampled1 = self.upsample1(pooled1)
        activated1 = self.activation(upsampled1) # Apply activation *before* W1
        merge1 = self.W1(activated1) # S1

        pooled2 = self.pool2(out2)
        upsampled2 = self.upsample2(pooled2)
        activated2 = self.activation(upsampled2) # Apply activation *before* W2
        merge2 = self.W2(activated2) # S2

        # Combine attention weights (S)
        out3 = self.add([merge1, merge2]) # Element-wise addition S = S1 + S2

        # Apply attention: Multiply main path by attention weights and add shortcut
        # Y = F'(X) * S + F''(X)
        y = self.multiply([out1, out3]) # F'(X) * S
        y = self.add([y, out2]) # Add shortcut connection F''(X)
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
        self.out_channels = filters // 4 # Output channels for SAM/CAM in AS-Net paper
        self.reduction_ratio = reduction_ratio

        # Convolution layers using compute dtype
        compute_dtype = mixed_precision.global_policy().compute_dtype
        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", kernel_initializer="he_normal", dtype=compute_dtype, name='conv4') # Branch for attention weights (shortcut path)

        # Channel attention mechanism
        self.gpool = GlobalAveragePooling2D(name='global_avg_pool')
        # Dense layers for channel attention weights (use compute dtype)
        self.fc1 = Dense(self.out_channels // self.reduction_ratio, activation="relu", use_bias=False, dtype=compute_dtype, name='fc1')
        self.fc2 = Dense(self.out_channels, activation="sigmoid", use_bias=False, dtype=compute_dtype, name='fc2')
        # Reshape needed to multiply with conv output (Batch, H, W, C) -> (Batch, 1, 1, C)
        self.reshape = Reshape((1, 1, self.out_channels), name='reshape')

        self.multiply = Multiply(name='multiply')
        self.add = Add(name='add') # Use Add layer

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs))) # Main feature path F'(X)
        out2 = self.conv4(inputs) # Shortcut path F''(X)

        # Calculate channel attention weights (C) from shortcut path F''(X)
        pooled = self.gpool(out2) # Global Average Pooling -> (Batch, C_out)
        fc1_out = self.fc1(pooled)
        fc2_out = self.fc2(fc1_out)
        out3 = self.reshape(fc2_out) # Reshape to (Batch, 1, 1, C_out) for broadcasting

        # Apply attention: Multiply main path by channel attention weights and add shortcut
        # Y = F'(X) * C + F''(X)
        y = self.multiply([out1, out3]) # Broadcasting applies channel weights F'(X) * C
        y = self.add([y, out2]) # Add shortcut connection F''(X)
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
        # Use tf.Variable for learnable weights
        self.alpha = tf.Variable(alpha_init, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(beta_init, trainable=True, name="beta", dtype=tf.float32)

        # 1x1 Convolution after weighted sum, followed by BN
        compute_dtype = mixed_precision.global_policy().compute_dtype
        # Output should be 1 channel before the final sigmoid in AS_Net
        self.conv = Conv2D(1, 1, padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv')
        self.bn = BatchNormalization(name='bn')
        self.add = Add(name='add_weighted') # Use Add layer

    def call(self, inputs):
        x, y = inputs # Expecting a tuple/list of (SAM_output, CAM_output)

        # Cast learnable weights to the compute dtype for multiplication
        compute_dtype = x.dtype # Get dtype from input tensor
        alpha_casted = tf.cast(self.alpha, compute_dtype)
        beta_casted = tf.cast(self.beta, compute_dtype)

        # Weighted sum: alpha * SAM_output + beta * CAM_output
        weighted_sum = self.add([alpha_casted * x, beta_casted * y])

        # Apply Conv -> BN
        convolved = self.conv(weighted_sum)
        bn_out = self.bn(convolved)
        # No activation here, final sigmoid is in the main model output layer
        return bn_out # Output has 1 channel

    def get_config(self):
        config = super(Synergy, self).get_config()
        # Store initial values, actual values are saved in weights
        config.update({"alpha_init": 0.5, "beta_init": 0.5})
        return config
# -

# ## Loss Functions

# +
class DiceLoss(Loss):
    """Computes the Dice Loss, a common metric for segmentation tasks."""
    def __init__(self, smooth=1e-6, name='dice_loss', **kwargs):
        # super(DiceLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.AUTO, **kwargs) # OLD
        super(DiceLoss, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs) # NEW
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype) # Ensure same dtype

        # Flatten tensors to calculate Dice score using tf.reshape
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate intersection and union using tf.reduce_sum instead of backend.sum
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Return Dice Loss
        return 1.0 - dice_coef

    def get_config(self):
        config = super(DiceLoss, self).get_config()
        config.update({"smooth": self.smooth})
        return config


class WBCE(Loss):
    """Weighted Binary Cross-Entropy Loss."""
    def __init__(self, weight=1.0, name='weighted_bce_loss', **kwargs):
        # super(WBCE, self).__init__(name=name, reduction=tf.keras.losses.Reduction.AUTO, **kwargs) # OLD
        super(WBCE, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs) # NEW
        self.weight = tf.cast(weight, tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        # Clip predictions to avoid log(0)
        epsilon_ = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

        # Calculate weighted BCE using tf.nn.weighted_cross_entropy_with_logits
        # Since model output is sigmoid, convert back to logits for the TF function
        logits = tf.math.log(y_pred / (1.0 - y_pred))

        # TF function expects targets (y_true) and logits
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=self.weight # This is where the weight is applied
        )

        # The function returns loss per element, so reduce it (e.g., mean)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super(WBCE, self).get_config()
        config.update({"weight": self.weight.numpy()}) # Store numpy value in config
        return config

# Alternative Loss Class using Focal Loss:
class CombinedFocalDiceLoss(Loss):
    """Combines Dice Loss and Binary Focal Crossentropy."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, focal_gamma=2.0, focal_alpha=None, name='combined_focal_dice_loss', **kwargs):
        # super().__init__(name=name, reduction=tf.keras.losses.Reduction.AUTO, **kwargs) # OLD
        super().__init__(name=name, reduction='sum_over_batch_size', **kwargs)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        # Use alpha for class weighting in focal loss if needed (e.g., 0.25 for positive class)
        # apply_class_balancing=True can be problematic, manual alpha is often better
        self.focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
            #apply_class_balancing=True, # May not be stable, use alpha instead if needed
            alpha=focal_alpha, # e.g., 0.25 or calculate based on imbalance
            gamma=focal_gamma,
            from_logits=False # Model output is sigmoid
        )
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        focal_loss_val = self.focal_loss(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)
        combined = (self.focal_weight * focal_loss_val) + (self.dice_weight * dice_loss_val)
        return combined

    def get_config(self):
        config = super(CombinedFocalDiceLoss, self).get_config()
        config.update({
            "focal_weight": self.focal_weight,
            "dice_weight": self.dice_weight,
            "focal_gamma": self.focal_loss.gamma,
            "focal_alpha": self.focal_loss.alpha
        })
        return config

# Combined Loss (Example: Dice + Weighted BCE)
class CombinedLoss(Loss):
    """Combines Dice Loss and Weighted Binary Cross-Entropy."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, class_weight=1.0, name='combined_loss', **kwargs):
        # super(CombinedLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.AUTO, **kwargs) # OLD
        super(CombinedLoss, self).__init__(name=name, reduction='sum_over_batch_size', **kwargs) # NEW
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wbce = WBCE(weight=class_weight)
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wbce = WBCE(weight=class_weight) # Instantiate WBCE with class weight
        self.dice_loss = DiceLoss()          # Instantiate Dice Loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        # Calculate individual losses
        bce_loss_val = self.wbce(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)

        # Combine losses with weights
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
        # Keras docs recommend using assign(0.)
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

# ## Data Preparation for BraTS

# +
def prepare_brats_data_gpu(
    metadata_file=METADATA_FILE,
    h5_dir=H5_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    modality_indices=MODALITY_INDICES, # Indices to load
    rgb_mapping_indices=RGB_MAPPING_INDICES, # How loaded modalities map to RGB
    num_modalities_loaded=NUM_MODALITIES_LOADED,
    input_channels=INPUT_CHANNELS, # Should be 3 for VGG16
    validation_split=0.2,
    random_seed=42,
):
    """
    Prepares the BraTS 2020 dataset from H5 slices using tf.data.
    Optimized for memory and GPU processing. Assumes H5 files contain slices.
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

    # Construct full path and check existence
    df['full_path'] = df['slice_path'].apply(lambda x: os.path.join(h5_dir, os.path.basename(x)))
    df = df[df['full_path'].apply(os.path.exists)].copy() # Filter out rows where the H5 file doesn't exist

    if df.empty:
        raise ValueError(f"No valid H5 files found based on metadata in {h5_dir}. Check paths.")

    print(f"Found {len(df)} existing H5 files referenced in metadata.")

    # Shuffle and split data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - validation_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Get lists of file paths
    train_files = train_df["full_path"].tolist()
    val_files = val_df["full_path"].tolist()

    # Function to parse a single H5 file using tf.py_function
    def parse_h5_file(file_path):
        def _parse_h5(path_tensor):
            path = path_tensor.numpy().decode("utf-8")
            try:
                with h5py.File(path, "r") as hf:
                    image_data = hf["image"][()] # Shape (H, W, 4) - float64
                    mask_data = hf["mask"][()]   # Shape (H, W, 3) - uint8

                    # --- Select Modalities ---
                    # Select the specified modalities (e.g., T1ce, FLAIR)
                    selected_modalities = image_data[:, :, modality_indices].astype(np.float32) # Shape (H, W, num_modalities_loaded)

                    # --- Create Binary Mask ---
                    # Combine all tumor classes (NCR/NET=1, ED=2, ET=4) into a single binary mask
                    # Assumes mask channels are [NCR/NET, ED, ET]
                    # Check mask channel interpretation if different
                    binary_mask = np.logical_or.reduce((mask_data[:, :, 0] > 0,
                                                        mask_data[:, :, 1] > 0,
                                                        mask_data[:, :, 2] > 0)).astype(np.float32) # Shape (H, W)

                    # Debug: Print shapes and types after loading
                    # print(f"Loaded {os.path.basename(path)}: Image shape {selected_modalities.shape}, Mask shape {binary_mask.shape}")
                    # Check mask values
                    # unique_mask_vals = np.unique(binary_mask)
                    # if not np.all(np.isin(unique_mask_vals, [0, 1])):
                    #     print(f"WARNING: Non-binary mask values found in {os.path.basename(path)}: {unique_mask_vals}")

                    return selected_modalities, binary_mask

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                # Return dummy data of expected original shape if loading fails
                original_h, original_w = 240, 240 # Assuming original size
                return (
                    np.zeros((original_h, original_w, num_modalities_loaded), dtype=np.float32),
                    np.zeros((original_h, original_w), dtype=np.float32),
                )

        # Wrap the Python function
        image, mask = tf.py_function(
            _parse_h5, [file_path], [tf.float32, tf.float32]
        )

        # --- Set Shapes ---
        # Explicitly set shapes after py_function, crucial for tf.data pipeline
        original_h, original_w = 240, 240 # Set expected original dimensions
        image.set_shape([original_h, original_w, num_modalities_loaded])
        mask.set_shape([original_h, original_w])

        return image, mask

    # Function to preprocess image and mask tensors
    def preprocess(image, mask):
        # --- Normalization ---
        # Apply Z-score normalization per channel (modality)
        normalized_channels = []
        for i in range(num_modalities_loaded):
            channel = image[:, :, i]
            mean = tf.reduce_mean(channel)
            std = tf.math.reduce_std(channel)
            # Add epsilon to prevent division by zero if a channel is constant
            normalized_channel = (channel - mean) / (std + 1e-8)
            normalized_channels.append(normalized_channel)
        image_normalized = tf.stack(normalized_channels, axis=-1) # Shape: (H, W, num_modalities_loaded)

        # --- RGB Mapping ---
        # Create 3-channel image for VGG16 by mapping/duplicating normalized modalities
        # Example: T1ce -> R, FLAIR -> G, T1ce -> B
        rgb_channels = [image_normalized[:, :, idx] for idx in rgb_mapping_indices]
        image_rgb = tf.stack(rgb_channels, axis=-1) # Shape: (H, W, 3)

        # --- Resizing ---
        # Resize image using bilinear interpolation
        image_resized = tf.image.resize(image_rgb, target_size, method='bilinear')
        # Resize mask using nearest neighbor to preserve binary values
        # Add channel dim to mask for resize, then remove
        mask_expanded = tf.expand_dims(mask, axis=-1)
        mask_resized = tf.image.resize(mask_expanded, target_size, method='nearest')
        mask_resized = tf.squeeze(mask_resized, axis=-1)

        # --- Final Mask Processing ---
        # Ensure mask is binary (0 or 1) after resizing
        mask_final = tf.cast(mask_resized > 0.5, tf.float32)
        # Add channel dimension to mask for consistency with model output shape (H, W, 1)
        mask_final = tf.expand_dims(mask_final, axis=-1)

        # --- Set Final Shapes ---
        image_resized.set_shape([target_size[0], target_size[1], input_channels])
        mask_final.set_shape([target_size[0], target_size[1], 1])

        # Debug: Print shapes and types during preprocessing
        # tf.print("Preprocessed Image Shape:", tf.shape(image_resized), "Mask Shape:", tf.shape(mask_final))
        # tf.print("Image dtype:", image_resized.dtype, "Mask dtype:", mask_final.dtype)
        # tf.print("Unique mask values after preprocess:", tf.unique(tf.reshape(mask_final, [-1]))[0])

        return image_resized, mask_final

    # --- Create tf.data Datasets ---
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_files)

    # --- Apply Transformations ---
    # Use AUTOTUNE for parallel calls
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA # Or FILE

    train_dataset = (
        train_dataset.with_options(options)
        .map(parse_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size) # Shuffle after loading and preprocessing
        .batch(batch_size)    # Batch data
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
    # print("Train Dataset Element Spec:", train_dataset.element_spec)
    # print("Validation Dataset Element Spec:", val_dataset.element_spec)

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
            print(f"Image value range: Min={tf.reduce_min(images):.4f}, Max={tf.reduce_max(images):.4f}")
            print(f"Mask value range: Min={tf.reduce_min(masks):.4f}, Max={tf.reduce_max(masks):.4f}")
            print(f"Unique mask values in batch: {tf.unique(tf.reshape(masks, [-1]))[0]}")


            for i in range(min(num_samples, num_in_batch)):
                plot_count += 1
                img = images[i].numpy()
                mask = masks[i].numpy().squeeze() # Remove channel dim for plotting

                # --- Display individual channels and combined RGB ---
                # Assuming RGB mapping [T1ce, FLAIR, T1ce] -> Channels 0, 1, 2
                titles = ["Input R (T1ce)", "Input G (FLAIR)", "Input B (T1ce)", "RGB Input", "Ground Truth Mask"]
                channels_to_plot = [img[:, :, 0], img[:, :, 1], img[:, :, 2], img, mask]

                for j, item in enumerate(channels_to_plot):
                    ax = plt.subplot(num_samples, len(titles), i * len(titles) + j + 1)
                    ax.set_title(titles[j])

                    # Normalize individual channels for display if needed (using percentiles)
                    if j < 3: # Individual channels
                         p_low, p_high = np.percentile(item, [2, 98])
                         item_disp = np.clip((item - p_low) / (p_high - p_low + 1e-8), 0, 1) if (p_high - p_low) > 1e-7 else item
                         plt.imshow(item_disp, cmap='gray')
                    elif j == 3: # RGB image
                        # RGB image might already be normalized, display as is or clip
                        img_rgb_disp = np.clip(item, 0, 1) # Clip Z-score normalized data to [0,1] for display
                        # Check for NaN/Inf
                        if np.isnan(img_rgb_disp).any() or np.isinf(img_rgb_disp).any():
                             print(f"Warning: NaN/Inf found in RGB image display for sample {i}")
                             img_rgb_disp = np.nan_to_num(img_rgb_disp) # Replace NaN/Inf with 0
                        plt.imshow(img_rgb_disp)
                    else: # Mask
                        if np.isnan(item).any() or np.isinf(item).any():
                             print(f"Warning: NaN/Inf found in mask display for sample {i}")
                             item = np.nan_to_num(item) # Replace NaN/Inf with 0
                        plt.imshow(item, cmap='jet', vmin=0, vmax=1) # Use 'jet' or 'viridis' for mask

                    plt.axis("off")

        plt.tight_layout(pad=0.5) # Add padding
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
class ConciseProgressCallback(tf.keras.callbacks.Callback):
    """Logs progress concisely and performs garbage collection."""
    def __init__(self, log_frequency=1): # Log every epoch by default
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

            # Monitor Synergy weights
            try:
                synergy_layer = self.model.get_layer('synergy')
                alpha_val = synergy_layer.alpha.numpy()
                beta_val = synergy_layer.beta.numpy()
                print(f"    Synergy weights - alpha: {alpha_val:.4f}, beta: {beta_val:.4f}")
            except Exception as e:
                # Layer might not exist if model architecture changes
                pass # print(f"    Could not get Synergy weights: {e}")

        # Force garbage collection at end of epoch
        gc.collect()

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n--- Training Finished ---")
        print(f"Total training time: {total_time:.2f} seconds")


# Learning Rate Scheduler Function
def lr_step_decay(epoch, lr):
    """Applies step decay to the learning rate."""
    initial_lr = LEARNING_RATE # Use the initial LEARNING_RATE constant
    drop = 0.5       # Factor to drop learning rate by
    epochs_drop = 10  # Number of epochs before dropping LR
    new_lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    # Add a minimum learning rate limit
    final_lr = max(new_lr, 1e-7)
    # print(f"Epoch {epoch+1}: LR = {final_lr:.7f}") # Optional: print LR each epoch
    return final_lr
# -

# ## Training Function

def train_model_distributed(
    dataset_func=prepare_brats_data_gpu,
    model_func=AS_Net,
    strategy=strategy, # Pass the distribution strategy
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_best_path=CHECKPOINT_BEST_PATH,
    output_dir=OUTPUT_DIR,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    input_channels=INPUT_CHANNELS,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    combined_loss_weights=COMBINED_LOSS_WEIGHTS, # Use constant from top
    # Alternative: Use Focal Loss
    # use_focal_loss=USE_FOCAL_LOSS,
    metrics=['binary_accuracy',
             DiceCoefficient(name='dice_coef'),
             IoU(name='iou'),
             tf.keras.metrics.Precision(thresholds=THRESHOLD, name="precision"), # Add Precision/Recall here
             tf.keras.metrics.Recall(thresholds=THRESHOLD, name="recall")],
):
    """Trains the AS-Net model using tf.distribute.Strategy."""

    # Calculate global batch size
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    print(f"--- Starting Training ---")
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

    # 2. Build and Compile Model within Strategy Scope
    with strategy.scope():
        print("Building model...")
        model = model_func(input_size=(img_height, img_width, input_channels))
        print("Model built.")

        print("Compiling model...")
        # Instantiate loss function within strategy scope
        # if use_focal_loss:
        #      print("Using Combined Focal + Dice Loss")
        #      loss_instance = CombinedFocalDiceLoss(
        #           focal_weight=combined_loss_weights['focal_weight'],
        #           dice_weight=combined_loss_weights['dice_weight'],
        #           focal_gamma=combined_loss_weights['focal_gamma']
        #      )
        # else:
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
            model.load_weights(latest_checkpoint).expect_partial() # Use expect_partial for optimizer state flexibility
            # Extract epoch number from checkpoint filename if possible (e.g., 'ckpt-10.h5')
            try:
                 # Assuming format like model_.10-0.123456.h5 or ckpt-10.h5 etc.
                 filename = os.path.basename(latest_checkpoint)
                 epoch_str = filename.split('-')[0].split('.')[-1] # Try common patterns
                 if not epoch_str.isdigit(): # Try another pattern if first fails
                      epoch_str = filename.split('-')[0].split('_')[-1]
                 initial_epoch = int(epoch_str)
                 print(f"Successfully loaded weights. Starting from epoch {initial_epoch + 1}")
            except Exception as parse_err:
                 print(f"Warning: Could not determine epoch number from checkpoint name '{latest_checkpoint}', starting from epoch 0. Error: {parse_err}")
                 initial_epoch = 0 # Default if parsing fails
        except Exception as load_err:
            print(f"Error loading weights from {latest_checkpoint}: {load_err}. Starting training from scratch.")
            initial_epoch = 0

    else:
        print("No checkpoint found, starting training from scratch.")


    # 3. Define Callbacks
    callbacks = [
        # Save weights only, save best based on val_dice_coef (higher is better)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_best_path,
            save_weights_only=True,
            monitor='val_dice_coef', # Monitor Dice Coefficient
            mode='max',             # Save model with highest Dice
            save_best_only=True,
            verbose=1
        ),
        # Save weights every epoch (for resuming)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, # Overwrites each epoch
            save_weights_only=True,
            save_freq='epoch',      # Save at the end of every epoch
            verbose=0
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',     # Monitor validation loss
            factor=0.5,             # Reduce LR by half
            patience=5,             # Wait 5 epochs with no improvement
            min_lr=1e-7,            # Minimum learning rate
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',     # Monitor validation loss
            patience=15,            # Wait 15 epochs with no improvement
            restore_best_weights=True, # Restore weights from best epoch (based on val_loss)
            verbose=1
        ),
        # Apply learning rate schedule
        tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=0),
        # Custom progress logger
        ConciseProgressCallback(log_frequency=1),
        # TensorBoard (optional)
        # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=1)
    ]

    # 4. Train the Model
    print(f"Starting training loop for {num_epochs - initial_epoch} potential epochs (may stop early)...")
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
        print(f"Error saving training history or plotting: {e}")

    # Force cleanup after training
    print("Cleaning up resources after training attempt...")
    del train_dataset, val_dataset # Explicitly delete datasets
    gc.collect()
    backend.clear_session() # Clear Keras session
    print("Cleaned up datasets and session.")

    return history, model # Return the final model state (might be from early stopping)

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

        # Determine available metrics (handle potential missing keys)
        metrics_to_plot = {'loss': 'Loss'}
        # Check for custom metrics and standard ones
        for key in history_dict.keys():
            if key.startswith('val_'): continue # Skip val metrics here
            if key == 'loss': continue
            if key == 'dice_coef': metrics_to_plot['dice_coef'] = 'Dice Coefficient'
            elif key == 'iou': metrics_to_plot['iou'] = 'IoU (Jaccard)'
            elif key == 'binary_accuracy': metrics_to_plot['binary_accuracy'] = 'Binary Accuracy'
            elif key == 'precision': metrics_to_plot['precision'] = 'Precision'
            elif key == 'recall': metrics_to_plot['recall'] = 'Recall'
            # Add other potential metrics if needed

        num_plots = len(metrics_to_plot)
        if num_plots == 0:
            print("No metrics found to plot (excluding loss).")
            return

        plt.figure(figsize=(6 * num_plots, 5))

        plot_index = 1
        for metric, title in metrics_to_plot.items():
            plt.subplot(1, num_plots, plot_index)
            # Plot training metric
            if metric in history_dict:
                plt.plot(epochs, history_dict[metric], 'bo-', label=f'Training {title}')
            else:
                 print(f"Warning: Training metric '{metric}' not found in history.")

            # Plot validation metric
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                plt.plot(epochs, history_dict[val_metric], 'ro-', label=f'Validation {title}')
            else:
                 print(f"Warning: Validation metric '{val_metric}' not found in history.")


            plt.title(f'Training and Validation {title}')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            # Set y-axis limits appropriately
            if metric != 'loss':
                 plt.ylim([0, 1]) # Limit Dice, IoU, Acc, etc. to [0, 1]
            plt.legend()
            plt.grid(True)
            plot_index += 1

        plt.tight_layout(pad=1.0)
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
    checkpoint_best_path=CHECKPOINT_BEST_PATH,
    output_folder=OUTPUT_DIR,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    input_channels=INPUT_CHANNELS,
    batch_size=BATCH_SIZE, # Use evaluation batch size
    dataset_func=prepare_brats_data_gpu,
    threshold=THRESHOLD,
    num_examples_to_save=5,
    loss_config=COMBINED_LOSS_WEIGHTS, # Pass loss config
    # use_focal_loss=USE_FOCAL_LOSS, # Pass loss choice
):
    """Evaluates the trained AS-Net model on the validation set."""
    print("\n--- Starting Model Evaluation ---")
    evaluation_results = None # Initialize results

    try:
        # 1. Load Validation Data
        # Use global batch size for evaluation dataset as well
        global_eval_batch_size = batch_size * strategy.num_replicas_in_sync
        print(f"Loading validation data with batch size: {global_eval_batch_size}...")
        _, val_dataset = dataset_func(batch_size=global_eval_batch_size, validation_split=0.2) # Ensure consistent split
        print("Validation dataset loaded.")

        # 2. Load or Use Provided Model
        if model is None:
             print(f"Loading model weights from best checkpoint: {checkpoint_best_path}")
             if not os.path.exists(checkpoint_best_path):
                  print(f"Error: Best checkpoint file not found at {checkpoint_best_path}.")
                  # Try loading the last epoch checkpoint instead
                  last_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
                  if not last_checkpoint:
                      print(f"Error: No checkpoint found in {os.path.dirname(checkpoint_path)}. Cannot evaluate.")
                      return None # Cannot proceed without a model
                  else:
                      print(f"Warning: Best checkpoint not found. Loading last epoch checkpoint: {last_checkpoint}")
                      checkpoint_to_load = last_checkpoint
             else:
                  checkpoint_to_load = checkpoint_best_path

             # Build model architecture again before loading weights
             print("Rebuilding model architecture for evaluation...")
             with strategy.scope(): # Load model within strategy scope if needed
                 model_eval = AS_Net(input_size=(img_height, img_width, input_channels))

                 # Compile is necessary to run evaluate, use the same loss/metrics as training
                 print("Compiling evaluation model...")
                 # Instantiate loss function
                 # if use_focal_loss:
                 #      print("Using Combined Focal + Dice Loss for eval compilation")
                 #      loss_instance = CombinedFocalDiceLoss(
                 #           focal_weight=loss_config['focal_weight'],
                 #           dice_weight=loss_config['dice_weight'],
                 #           focal_gamma=loss_config['focal_gamma']
                 #      )
                 # else:
                 print("Using Combined WBCE + Dice Loss for eval compilation")
                 loss_instance = CombinedLoss(
                      bce_weight=loss_config['bce_weight'],
                      dice_weight=loss_config['dice_weight'],
                      class_weight=loss_config['class_weight']
                 )
                 # Instantiate optimizer (not used for eval weights, but needed for compile)
                 optimizer = tf.keras.optimizers.Adam()
                 # Apply loss scaling if needed
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
                 model_eval.load_weights(checkpoint_to_load).expect_partial() # Use expect_partial
                 print(f"Successfully loaded weights into new model instance from {checkpoint_to_load}")
        else:
            print("Using provided trained model instance for evaluation.")
            # Ensure the provided model is compiled with the necessary metrics if not already done
            if not model.optimizer:
                 print("Compiling the provided model for evaluation...")
                 with strategy.scope():
                      # Compile similarly to how it's done when loading from checkpoint
                      # Instantiate loss function
                      # if use_focal_loss: loss_instance = CombinedFocalDiceLoss(**loss_config)
                      # else: loss_instance = CombinedLoss(**loss_config)
                      loss_instance = CombinedLoss(**loss_config) # Assuming WBCE+Dice if not focal

                      optimizer = tf.keras.optimizers.Adam()
                      if mixed_precision.global_policy().name == 'mixed_float16': optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                      model.compile(
                           optimizer=optimizer, loss=loss_instance,
                           metrics=['binary_accuracy', DiceCoefficient(name='dice_coef'), IoU(name='iou'),
                                    tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
                                    tf.keras.metrics.Recall(thresholds=threshold, name="recall")]
                      )
                      print("Provided model compiled.")
            model_eval = model # Use the model passed from training

        # 3. Evaluate using model.evaluate()
        print("Evaluating model on validation set...")
        evaluation_results = model_eval.evaluate(val_dataset, verbose=1, return_dict=True)
        print("\nKeras Evaluation Results:")
        for name, value in evaluation_results.items():
            print(f"- {name}: {value:.4f}")

        # 4. Calculate F1 Score (using precision and recall from Keras metrics)
        precision_val = evaluation_results.get('precision', 0.0)
        recall_val = evaluation_results.get('recall', 0.0)
        if (precision_val + recall_val) > 1e-7: # Avoid division by zero
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        else:
            f1_val = 0.0
        evaluation_results['f1_score'] = f1_val # Add F1 to the results dictionary
        print(f"- f1_score: {f1_val:.4f} (calculated from Precision/Recall)")

        # 5. Save Performance Metrics to File
        try:
            perf_file_path = os.path.join(output_folder, "performances.txt")
            with open(perf_file_path, "w") as file_perf:
                file_perf.write("Evaluation Metrics:\n")
                file_perf.write("-------------------\n")
                file_perf.write(f"Loss: {evaluation_results.get('loss', 'N/A'):.4f}\n")
                file_perf.write(f"Binary Accuracy: {evaluation_results.get('binary_accuracy', 'N/A'):.4f}\n")
                file_perf.write(f"Dice Coefficient: {evaluation_results.get('dice_coef', 'N/A'):.4f}\n")
                file_perf.write(f"IoU (Jaccard): {evaluation_results.get('iou', 'N/A'):.4f}\n")
                file_perf.write(f"Precision (Threshold={threshold}): {evaluation_results.get('precision', 'N/A'):.4f}\n")
                file_perf.write(f"Recall/Sensitivity (Threshold={threshold}): {evaluation_results.get('recall', 'N/A'):.4f}\n")
                file_perf.write(f"F1-Score (Threshold={threshold}): {evaluation_results.get('f1_score', 'N/A'):.4f}\n")
            print(f"Evaluation results saved to {perf_file_path}")
        except Exception as e:
            print(f"Error saving performance metrics to file: {e}")

        # 6. Generate and Save Prediction Examples
        print("\nGenerating prediction examples...")
        # Ensure the model used for prediction is the evaluated one
        save_prediction_examples(model_eval, val_dataset, output_folder, num_examples=num_examples_to_save, threshold=threshold)

        print("--- Evaluation Finished ---")
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
        # Avoid clearing session if model was passed in and might be used later
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
            predictions = model.predict(images)
            # Apply threshold to get binary predictions
            binary_predictions = tf.cast(predictions >= threshold, tf.float32).numpy()
            predictions = predictions.numpy() # Get probability maps as numpy array
            images = images.numpy()
            masks = masks.numpy()

            print("Plotting and saving examples...")
            for j in range(min(num_examples, images.shape[0])):
                plt.figure(figsize=(16, 4)) # Width adjusted for 4 plots

                # --- Plot 1: Original RGB Input ---
                plt.subplot(1, 4, 1)
                plt.title("Input RGB")
                # Clip Z-score normalized data to [0,1] for display
                img_display = np.clip(images[j], 0, 1)
                if np.isnan(img_display).any() or np.isinf(img_display).any():
                     print(f"Warning: NaN/Inf found in input image display for example {j}")
                     img_display = np.nan_to_num(img_display)
                plt.imshow(img_display)
                plt.axis("off")

                # --- Plot 2: Ground Truth Mask ---
                plt.subplot(1, 4, 2)
                plt.title("Ground Truth Mask")
                # Display background image slightly faded
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                # Overlay mask
                gt_mask_display = masks[j].squeeze()
                if np.isnan(gt_mask_display).any() or np.isinf(gt_mask_display).any():
                     print(f"Warning: NaN/Inf found in GT mask display for example {j}")
                     gt_mask_display = np.nan_to_num(gt_mask_display)
                plt.imshow(gt_mask_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1) # Use squeeze for mask (H,W)
                plt.axis("off")

                # --- Plot 3: Prediction Probability Map ---
                plt.subplot(1, 4, 3)
                plt.title("Prediction Probabilities")
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                # Overlay probability map
                pred_prob_display = predictions[j].squeeze()
                if np.isnan(pred_prob_display).any() or np.isinf(pred_prob_display).any():
                     print(f"Warning: NaN/Inf found in probability map display for example {j}")
                     pred_prob_display = np.nan_to_num(pred_prob_display)
                prob_map = plt.imshow(pred_prob_display, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")
                # Add colorbar for probability map
                # plt.colorbar(prob_map, fraction=0.046, pad=0.04) # Optional: Add colorbar

                # --- Plot 4: Binary Prediction (Thresholded) ---
                plt.subplot(1, 4, 4)
                plt.title(f"Binary Prediction (t={threshold:.2f})")
                plt.imshow(img_display, cmap='gray', alpha=0.6)
                # Overlay binary prediction
                binary_pred_display = binary_predictions[j].squeeze()
                if np.isnan(binary_pred_display).any() or np.isinf(binary_pred_display).any():
                     print(f"Warning: NaN/Inf found in binary prediction display for example {j}")
                     binary_pred_display = np.nan_to_num(binary_pred_display)
                plt.imshow(binary_pred_display, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
                plt.axis("off")

                # Save the figure
                plt.tight_layout(pad=0.5) # Add padding
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
        # Calculate duration in seconds
        duration_seconds = time.time() - start_time
        # Convert to hours and minutes
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"

    try:
        with open(completion_file, "w") as f:
            f.write(f"VGG16 AS-Net Training Completed at: {timestamp}\n\n")
            f.write("Training Configuration:\n")
            f.write(f"- Model: AS-Net with VGG16 encoder\n")
            f.write(f"- Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}\n")
            f.write(f"- Input Channels: {INPUT_CHANNELS}\n")
            f.write(f"- Batch size (per replica): {BATCH_SIZE}\n")
            f.write(f"- Global Batch size: {BATCH_SIZE * strategy.num_replicas_in_sync}\n")
            f.write(f"- Epochs planned: {NUM_EPOCHS}\n")
            f.write(f"- Initial Learning rate: {LEARNING_RATE}\n")
            f.write(f"- Mixed Precision Policy: {mixed_precision.global_policy().name}\n")
            f.write(f"- Loss Config: {COMBINED_LOSS_WEIGHTS}\n")
            f.write(f"- Total Duration: {duration_str}\n\n")
            # f.write(f"- Using Focal Loss: {USE_FOCAL_LOSS}\n\n") # If using focal loss toggle

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
# Check if completion file exists to skip training
if os.path.exists(COMPLETION_FILE):
     print(f"Completion file '{COMPLETION_FILE}' found. Skipping training.")
     # Ensure the model architecture is defined even if skipping training,
     # so evaluation can build it before loading weights.
     # Build model within strategy scope
     with strategy.scope():
          model = AS_Net(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS))
          print("Model architecture defined for potential evaluation.")
     history = None # No history object if training is skipped
else:
     print("Completion file not found. Starting training process...")
     history, model = train_model_distributed() # history and model are returned

# Step 2: Evaluate the model
# Pass the model ONLY if training actually ran and returned a model
model_to_evaluate = model if 'model' in locals() and model is not None else None
evaluation_results = evaluate_model(model=model_to_evaluate) # Pass model if it exists

# Step 3: Create completion notification (will include eval results if successful)
create_completion_notification(start_time=script_start_time)

# Final cleanup
print("\n--- Final Script Cleanup ---")
if 'model' in locals(): del model
if 'train_dataset' in locals(): del train_dataset
if 'val_dataset' in locals(): del val_dataset
if 'evaluation_results' in locals(): del evaluation_results
if 'history' in locals(): del history
gc.collect()
backend.clear_session()
print("Script execution completed successfully!")
