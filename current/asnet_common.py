# asnet_common.py
# Contains shared components for AS-Net implementations

import os
import gc
import time
import math
import numpy as np
import tensorflow as tf
from keras import Model, backend
from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,  # Keep for SAM module
    UpSampling2D,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Add,
    Layer,  # For ResizeToMatchLayer
)
from keras.losses import Loss
from keras.metrics import Metric
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tensorflow.keras import mixed_precision
import matplotlib.colors as mcolors  # Added for colormaps
import matplotlib.patches as mpatches  # Added for legends
from tensorflow.keras.callbacks import TensorBoard
import datetime

# --- GPU Configuration and Mixed Precision ---


def setup_gpu_and_mixed_precision(batch_size_per_replica, use_mixed_precision=False):
    """Configures GPU memory growth, distribution strategy, and mixed precision."""
    print("--- GPU Configuration ---")
    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.get_strategy()  # Default strategy

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(
                f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

            if len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(
                    f"Running on {strategy.num_replicas_in_sync} GPU(s) using MirroredStrategy.")
            else:
                strategy = tf.distribute.get_strategy()  # Default for single GPU
                print("Running on single GPU (default strategy).")

        except RuntimeError as e:
            print(
                f"GPU Configuration Error: {e}. Falling back to default strategy.")
            strategy = tf.distribute.get_strategy()
            print("Running on CPU or single GPU (default strategy).")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy (CPU)
        print("No GPU detected. Running on CPU.")

    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print("Number of replicas in sync:", strategy.num_replicas_in_sync)
    print("Batch size per replica:", batch_size_per_replica)
    print("Global Batch Size (per replica * num replicas):", global_batch_size)

    print("\n--- Mixed Precision Configuration ---")
    policy_name = 'mixed_float16' if use_mixed_precision else 'float32'
    policy = mixed_precision.Policy(policy_name)
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {policy.name}")
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    # Optional JIT configuration (can uncomment if desired)
    # tf.config.optimizer.set_jit(True)
    # print("JIT compilation enabled.")

    return strategy, global_batch_size

# --- Custom Layer for Resizing (Needed for EfficientNet) ---


class ResizeToMatchLayer(Layer):
    """Resizes input tensor to match the spatial dimensions of the target tensor."""

    def __init__(self, name=None, **kwargs):
        super(ResizeToMatchLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        x_to_resize, target = inputs
        target_shape = tf.shape(target)
        target_height, target_width = target_shape[1], target_shape[2]
        return tf.image.resize(x_to_resize, [target_height, target_width], method='bilinear')

    def compute_output_shape(self, input_shape):
        # Output shape: (Batch, Target_H, Target_W, Channels_of_x_to_resize)
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3])

# --- AS-Net Core Modules (SAM, CAM, Synergy) ---


class SAM(Model):
    """Spatial Attention Module"""

    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        # Adjust reduction factor based on input filter size for robustness
        self.out_channels = max(
            16, filters // 8 if filters > 128 else filters // 4)
        compute_dtype = mixed_precision.global_policy().compute_dtype

        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')  # F'(X)
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv4')  # F''(X)

        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.upsample1 = UpSampling2D(
            (2, 2), interpolation="bilinear", name='upsample1')
        self.W1 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal",
                         dtype=compute_dtype, name='W1_map')  # Make output 1 channel

        self.pool2 = MaxPooling2D((4, 4), name='pool2')
        self.upsample2 = UpSampling2D(
            (4, 4), interpolation="bilinear", name='upsample2')
        self.W2 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal",
                         dtype=compute_dtype, name='W2_map')  # Make output 1 channel

        self.add_attention = Add(name='add_attention')
        self.multiply_attention = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')
        # Custom layer instance for resizing if needed, can be reused
        self.resize_layer = ResizeToMatchLayer(name=f'{name}_resize_attention')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))  # F'(X)
        out2 = self.conv4(inputs)  # F''(X)

        # Attention Branch 1
        pool1 = self.pool1(out2)
        up1 = self.upsample1(pool1)
        # Ensure spatial dimensions match F''(X) before applying W1
        up1 = self.resize_layer([up1, out2])
        att1 = self.W1(up1)  # S1 (B, H, W, 1)

        # Attention Branch 2
        pool2 = self.pool2(out2)
        up2 = self.upsample2(pool2)
        # Ensure spatial dimensions match F''(X) before applying W2
        up2 = self.resize_layer([up2, out2])
        att2 = self.W2(up2)  # S2 (B, H, W, 1)

        # Combine attention maps
        attention_map = self.add_attention(
            [att1, att2])  # S = S1 + S2 (B, H, W, 1)

        # Apply attention: Y = F'(X) * S + F''(X)
        # attention_map (B,H,W,1) is broadcast across channels of out1 (B,H,W,C_out)
        attended_features = self.multiply_attention(
            [out1, attention_map])  # F'(X) * S
        y = self.add_residual([attended_features, out2])  # Add F''(X)
        return y

    def get_config(self):
        config = super(SAM, self).get_config()
        config.update({"filters": self.filters})
        return config


class CAM(Model):
    """Channel Attention Module"""

    def __init__(self, filters, reduction_ratio=16, name='cam', **kwargs):
        super(CAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        # Adjust reduction factor based on input filter size
        self.out_channels = max(
            16, filters // 8 if filters > 128 else filters // 4)
        self.reduction_ratio = reduction_ratio
        compute_dtype = mixed_precision.global_policy().compute_dtype

        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')  # F'(X)
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv4')  # F''(X)

        self.gpool = GlobalAveragePooling2D(
            name='global_avg_pool', keepdims=True)  # Keep dims
        reduced_channels = max(1, self.out_channels // self.reduction_ratio)
        self.fc1 = Dense(reduced_channels, activation="relu",
                         use_bias=False, dtype=compute_dtype, name='fc1')
        self.fc2 = Dense(self.out_channels, activation="sigmoid", use_bias=False,
                         dtype=compute_dtype, name='fc2')  # Output (B, 1, 1, C_out)

        self.multiply_attention = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))  # F'(X)
        out2 = self.conv4(inputs)  # F''(X)

        # Channel attention from F''(X)
        pooled = self.gpool(out2)  # (B, 1, 1, C_out)
        fc1_out = self.fc1(pooled)
        channel_attention_weights = self.fc2(fc1_out)  # (B, 1, 1, C_out)

        # Apply attention: Y = F'(X) * C + F''(X)
        # Weights (B,1,1,C) broadcast across spatial dims of out1 (B,H,W,C)
        recalibrated_features = self.multiply_attention(
            [out1, channel_attention_weights])  # F'(X) * C
        y = self.add_residual([recalibrated_features, out2])  # Add F''(X)
        return y

    def get_config(self):
        config = super(CAM, self).get_config()
        config.update({"filters": self.filters,
                      "reduction_ratio": self.reduction_ratio})
        return config


class Synergy(Model):
    """Combines SAM and CAM outputs with learnable weights."""

    def __init__(self, alpha_init=0.5, beta_init=0.5, name='synergy', **kwargs):
        super(Synergy, self).__init__(name=name, **kwargs)
        # Use float32 for stability of learnable weights
        self.alpha = tf.Variable(
            alpha_init, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(beta_init, trainable=True,
                                name="beta", dtype=tf.float32)
        compute_dtype = mixed_precision.global_policy().compute_dtype

        # Output 1 channel for final segmentation map
        self.conv = Conv2D(
            1, 1, padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv')
        self.bn = BatchNormalization(name='bn')
        self.add = Add(name='add_weighted')

    def call(self, inputs):
        sam_features, cam_features = inputs
        compute_dtype = sam_features.dtype  # Match input dtype for operations
        alpha_casted = tf.cast(self.alpha, compute_dtype)
        beta_casted = tf.cast(self.beta, compute_dtype)

        # Weighted sum
        weighted_sum = self.add(
            [alpha_casted * sam_features, beta_casted * cam_features])

        # Conv -> BN (No activation here, final sigmoid is outside)
        convolved = self.conv(weighted_sum)
        bn_out = self.bn(convolved)
        return bn_out  # Shape (B, H, W, 1)

    def get_config(self):
        config = super(Synergy, self).get_config()
        # Store initial values; actual learned values are in weights
        config.update({"alpha_init": 0.5, "beta_init": 0.5})
        return config


# --- Loss Functions ---

class DiceLoss(Loss):
    """Computes the Dice Loss."""

    def __init__(self, smooth=1e-6, name='dice_loss', **kwargs):
        super(DiceLoss, self).__init__(
            name=name, reduction='sum_over_batch_size', **kwargs)
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
        super(WBCE, self).__init__(
            name=name, reduction='sum_over_batch_size', **kwargs)
        self.weight = tf.cast(weight, tf.float32)  # Store as float32

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon_ = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
        logits = tf.math.log(y_pred / (1.0 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            # Cast weight to logits dtype
            pos_weight=tf.cast(self.weight, logits.dtype)
        )
        # Framework handles reduction based on 'sum_over_batch_size'
        return loss

    def get_config(self):
        config = super(WBCE, self).get_config()
        config.update({"weight": self.weight.numpy()})  # Store numpy value
        return config


class CombinedLoss(Loss):
    """Combines Dice Loss and Weighted Binary Cross-Entropy."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, class_weight=1.0, name='combined_loss', **kwargs):
        super(CombinedLoss, self).__init__(
            name=name, reduction='sum_over_batch_size', **kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wbce = WBCE(weight=class_weight)
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        bce_loss_val = self.wbce(y_true, y_pred)
        dice_loss_val = self.dice_loss(y_true, y_pred)
        # Combine losses; framework handles reduction
        combined = (self.bce_weight * bce_loss_val) + \
            (self.dice_weight * dice_loss_val)
        return combined

    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "class_weight": self.wbce.weight.numpy()
        })
        return config


# --- Custom Metrics ---

class DiceCoefficient(Metric):
    """Computes the Dice Coefficient metric."""

    def __init__(self, threshold=0.5, smooth=1e-6, name='dice_coefficient', dtype=None):
        super(DiceCoefficient, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection_sum = self.add_weight(
            name='intersection_sum', initializer='zeros', dtype=tf.float32)
        self.union_sum = self.add_weight(
            name='union_sum', initializer='zeros', dtype=tf.float32)

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
        dice = (2.0 * self.intersection_sum + self.smooth) / \
            (self.union_sum + self.smooth)
        return tf.cast(dice, self._dtype) if self._dtype else dice

    def reset_state(self):
        self.intersection_sum.assign(0.0)
        self.union_sum.assign(0.0)

    def get_config(self):
        config = super(DiceCoefficient, self).get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


class IoU(Metric):
    """Computes the Intersection over Union (IoU) or Jaccard Index."""

    def __init__(self, threshold=0.5, smooth=1e-6, name='iou', dtype=None):
        super(IoU, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection_sum = self.add_weight(
            name='intersection_sum', initializer='zeros', dtype=tf.float32)
        self.union_sum = self.add_weight(
            name='union_sum', initializer='zeros', dtype=tf.float32)

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
        iou = (self.intersection_sum + self.smooth) / \
            (self.union_sum + self.smooth)
        return tf.cast(iou, self._dtype) if self._dtype else iou

    def reset_state(self):
        self.intersection_sum.assign(0.0)
        self.union_sum.assign(0.0)

    def get_config(self):
        config = super(IoU, self).get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


# --- Data Preparation ---

def prepare_brats_data_gpu(
    metadata_file,
    h5_dir,
    target_size,  # Tuple (height, width)
    batch_size,  # Global batch size
    buffer_size,
    modality_indices,
    rgb_mapping_indices,
    input_channels,  # Expected final channels (e.g., 3)
    validation_split=0.2,
    random_seed=42,
    # --- New ---
    # Index of modality to use for display (0:T1, 1:T1ce, 2:T2, 3:FLAIR)
    reference_modality_index=3,
):
    """
    Prepares the BraTS 2020 dataset from H5 slices using tf.data.
    Outputs:
    1. image_model_input: Z-scored, RGB-mapped, resized image for model training.
    2. mask_binary_gt: Binary (0/1), resized ground truth mask for loss calculation.
    3. image_display_ref: Reference modality (e.g., FLAIR), normalized [0,1], resized for visualization.
    4. mask_multiclass_gt: Multi-class (0,1,2,4), resized ground truth mask for visualization.
    """
    print("--- Setting up Data Pipeline (v2 - with display data) ---")
    print(f"Target image size: {target_size}")
    print(
        f"Reference modality for display: Index {reference_modality_index} ({['T1', 'T1ce', 'T2', 'FLAIR'][reference_modality_index]})")
    num_modalities_loaded = len(modality_indices)

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"H5 data directory not found at {h5_dir}")

    df = pd.read_csv(metadata_file)
    df['full_path'] = df['slice_path'].apply(
        lambda x: os.path.join(h5_dir, os.path.basename(x)))
    df = df[df['full_path'].apply(os.path.exists)].copy()

    if df.empty:
        raise ValueError(
            f"No valid H5 files found based on metadata in {h5_dir}. Check paths.")
    print(f"Found {len(df)} existing H5 files referenced in metadata.")

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - validation_split))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(
        f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_files = train_df["full_path"].tolist()
    val_files = val_df["full_path"].tolist()

    # --- Modified parse function ---
    def parse_h5_file(file_path):
        def _parse_h5(path_tensor):
            path = path_tensor.numpy().decode("utf-8")
            try:
                with h5py.File(path, "r") as hf:
                    # Load all 4 modalities for potential display use
                    image_all_modalities = hf["image"][()].astype(
                        np.float32)  # (H, W, 4)
                    # Load original 3-channel mask
                    mask_original_3ch = hf["mask"][()]  # (H, W, 3) uint8

                    # Select modalities needed for model input based on modality_indices
                    image_selected_modalities = image_all_modalities[:,
                                                                     :, modality_indices]

                    # Extract the reference modality for display
                    image_ref_modality = image_all_modalities[:,
                                                              :, reference_modality_index]

                    # Create binary mask (for loss)
                    mask_binary = np.logical_or.reduce((mask_original_3ch[:, :, 0] > 0,
                                                        mask_original_3ch[:,
                                                                          :, 1] > 0,
                                                        mask_original_3ch[:, :, 2] > 0)).astype(np.float32)  # (H, W)

                    # Create multi-class mask (for display) - using BraTS values 1, 2, 4
                    # NCR/NET = 1 (Channel 0), ED = 2 (Channel 1), ET = 4 (Channel 2)
                    mask_multiclass = np.zeros_like(
                        mask_binary, dtype=np.float32)
                    mask_multiclass[mask_original_3ch[:, :, 0]
                                    > 0] = 1  # NCR/NET
                    mask_multiclass[mask_original_3ch[:, :, 1] > 0] = 2  # ED
                    # ET (Enhancing Tumor comes last to overwrite others if overlapping)
                    mask_multiclass[mask_original_3ch[:, :, 2] > 0] = 4

                    return (image_selected_modalities, mask_binary,
                            image_ref_modality, mask_multiclass)
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                # Return dummy data with correct number of elements
                original_h, original_w = 240, 240
                dummy_selected = np.zeros(
                    (original_h, original_w, num_modalities_loaded), dtype=np.float32)
                dummy_binary = np.zeros(
                    (original_h, original_w), dtype=np.float32)
                dummy_ref = np.zeros(
                    (original_h, original_w), dtype=np.float32)
                dummy_multi = np.zeros(
                    (original_h, original_w), dtype=np.float32)
                return dummy_selected, dummy_binary, dummy_ref, dummy_multi

        # Update output types
        image_sel, mask_bin, img_ref, mask_multi = tf.py_function(
            _parse_h5, [file_path], [tf.float32,
                                     tf.float32, tf.float32, tf.float32]
        )

        # Set shapes (assuming original BraTS size)
        original_h, original_w = 240, 240
        image_sel.set_shape([original_h, original_w, num_modalities_loaded])
        mask_bin.set_shape([original_h, original_w])
        img_ref.set_shape([original_h, original_w])
        mask_multi.set_shape([original_h, original_w])

        return image_sel, mask_bin, img_ref, mask_multi

    # --- Modified preprocess function ---
    def preprocess(image_selected, mask_binary, image_ref, mask_multiclass):
        # 1. Process image for MODEL INPUT
        # Z-score selected modalities
        normalized_channels = []
        for i in range(num_modalities_loaded):
            channel = image_selected[:, :, i]
            mean = tf.reduce_mean(channel)
            std = tf.math.reduce_std(channel)
            normalized_channel = (channel - mean) / (std + 1e-8)
            normalized_channels.append(normalized_channel)
        image_normalized = tf.stack(normalized_channels, axis=-1)
        # Map to RGB
        rgb_channels = [image_normalized[:, :, idx]
                        for idx in rgb_mapping_indices]
        image_rgb = tf.stack(rgb_channels, axis=-1)
        # Resize
        image_model_input_resized = tf.image.resize(
            image_rgb, target_size, method='bilinear')
        # Cast (model-specific scaling like -1 to 1 happens inside model func)
        image_model_input = tf.cast(image_model_input_resized, tf.float32)

        # 2. Process BINARY mask for LOSS
        mask_binary_expanded = tf.expand_dims(mask_binary, axis=-1)
        mask_binary_resized = tf.image.resize(
            mask_binary_expanded, target_size, method='nearest')
        mask_binary_final = tf.cast(
            mask_binary_resized > 0.5, tf.float32)  # Ensure binary 0/1

        # 3. Process reference image for DISPLAY
        # Normalize reference modality to [0, 1] for consistent display
        min_val = tf.reduce_min(image_ref)
        max_val = tf.reduce_max(image_ref)
        image_ref_normalized = (image_ref - min_val) / \
            (max_val - min_val + 1e-8)
        # Resize reference image
        image_display_ref_resized = tf.image.resize(tf.expand_dims(image_ref_normalized, axis=-1),
                                                    target_size, method='bilinear')
        image_display_ref = tf.squeeze(
            image_display_ref_resized, axis=-1)  # Back to H, W

        # 4. Process MULTI-CLASS mask for DISPLAY
        mask_multiclass_expanded = tf.expand_dims(mask_multiclass, axis=-1)
        mask_multiclass_resized = tf.image.resize(
            mask_multiclass_expanded, target_size, method='nearest')
        # Ensure values are integers after resize (0, 1, 2, 4)
        mask_multiclass_final = tf.round(mask_multiclass_resized)
        # No need to squeeze, keep channel dim (H, W, 1) for consistency? Or squeeze? Let's squeeze for easier plotting.
        mask_multiclass_final = tf.squeeze(mask_multiclass_final, axis=-1)

        # Set Final Shapes
        image_model_input.set_shape(
            [target_size[0], target_size[1], input_channels])
        # Keep channel dim for loss
        mask_binary_final.set_shape([target_size[0], target_size[1], 1])
        image_display_ref.set_shape([target_size[0], target_size[1]])
        mask_multiclass_final.set_shape([target_size[0], target_size[1]])

        return image_model_input, mask_binary_final, image_display_ref, mask_multiclass_final

    # --- Create tf.data Datasets ---
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_files)

    # --- Apply Transformations ---
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = (
        train_dataset.with_options(options)
        .shuffle(buffer_size)
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
    print("Dataset Element Spec (Train):", train_dataset.element_spec)
    # Expected Output: ( TensorSpec(shape=(None, H, W, 3), dtype=tf.float32, name=None),       <- Model Input
    #                     TensorSpec(shape=(None, H, W, 1), dtype=tf.float32, name=None),       <- Binary Mask (Loss)
    #                     TensorSpec(shape=(None, H, W), dtype=tf.float32, name=None),       <- Ref Image (Display)
    #                     TensorSpec(shape=(None, H, W), dtype=tf.float32, name=None) )      <- Multi-Class Mask (Display)

    return train_dataset, val_dataset


def train_model_distributed(
    model_func,
    dataset_func,
    strategy,
    global_batch_size,
    # --- Constants to pass ---
    variant_name,
    checkpoint_path,
    checkpoint_best_path,
    output_dir,
    img_height,
    img_width,
    input_channels,
    num_epochs,
    initial_learning_rate,
    combined_loss_weights,
    metrics_list,
    threshold,
    metadata_file,
    h5_data_dir,
    buffer_size,
    modality_indices,
    rgb_mapping_indices,
    validation_split=0.2,
    lr_scheduler_drop=0.5,
    lr_scheduler_epochs_drop=10,
    reference_modality_index=3,  # Pass ref modality index to data func
    **model_kwargs
):
    """Trains the specified AS-Net model variant using tf.distribute.Strategy."""

    print(f"--- Starting Training ({variant_name}) ---")
    print(f"Target Image Size: {img_height}x{img_width}")
    print(f"Epochs: {num_epochs}")
    print(f"Initial Learning Rate: {initial_learning_rate}")
    print(f"Loss configuration: {combined_loss_weights}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Best Checkpoint path: {checkpoint_best_path}")

    # 1. Create Datasets (Pass reference_modality_index)
    print("Preparing datasets...")
    full_train_dataset, full_val_dataset = dataset_func(
        metadata_file=metadata_file,
        h5_dir=h5_data_dir,
        target_size=(img_height, img_width),
        batch_size=global_batch_size,
        buffer_size=buffer_size,
        modality_indices=modality_indices,
        rgb_mapping_indices=rgb_mapping_indices,
        input_channels=input_channels,
        validation_split=validation_split,
        reference_modality_index=reference_modality_index  # Pass argument
    )
    print("Full datasets prepared.")

    # Create datasets suitable for model.fit (input, binary_mask only)
    # IMPORTANT: This map operation selects the components needed for training
    train_dataset_for_fit = full_train_dataset.map(
        lambda img_in, mask_bin, img_ref, mask_multi: (img_in, mask_bin))
    val_dataset_for_fit = full_val_dataset.map(
        lambda img_in, mask_bin, img_ref, mask_multi: (img_in, mask_bin))
    print("Created datasets for model.fit (image_input, binary_mask).")
    print("Train Dataset Spec (for fit):", train_dataset_for_fit.element_spec)

    # ... existing code for model building, compilation, checkpoint loading ...
    with strategy.scope():
        print("Building model...")
        model = model_func(
            input_size=(img_height, img_width, input_channels),
            **model_kwargs  # Pass variant etc. if needed by the specific model func
        )
        print("Model built.")

        print("Compiling model...")
        loss_instance = CombinedLoss(
            bce_weight=combined_loss_weights['bce_weight'],
            dice_weight=combined_loss_weights['dice_weight'],
            class_weight=combined_loss_weights['class_weight']
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate)
        if mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            print("Loss scaling applied for mixed precision.")

        # Ensure metrics requiring threshold get it
        compiled_metrics = []
        for m in metrics_list:
            if isinstance(m, str):
                compiled_metrics.append(m)
            elif hasattr(m, 'get_config'):
                # Fix here: Handle Precision and Recall metrics specifically
                if m.__class__.__name__ == 'Precision' or m.__class__.__name__ == 'Recall':
                    config = m.get_config()
                    if 'threshold' in config:
                        # Convert singular 'threshold' to plural 'thresholds'
                        threshold_value = config.pop('threshold')
                        config['thresholds'] = threshold_value
                    compiled_metrics.append(m.__class__.from_config(config))
                else:
                    config = m.get_config()
                    if 'threshold' in config:
                        config['threshold'] = threshold
                    compiled_metrics.append(m.__class__.from_config(config))
            else:
                compiled_metrics.append(m)

        model.compile(
            optimizer=optimizer,
            loss=loss_instance,
            metrics=compiled_metrics
        )
        print("Model compiled.")
        model.summary(line_length=120)

    # Check for checkpoint resume
    latest_checkpoint = tf.train.latest_checkpoint(
        os.path.dirname(checkpoint_path))
    initial_epoch = 0
    if latest_checkpoint:
        # ... existing code for checkpoint loading ...
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        try:
            with strategy.scope():  # Load weights within scope
                model.load_weights(latest_checkpoint).expect_partial()
            # Simple epoch extraction (adjust if checkpoint naming differs significantly)
            try:
                # Try different common patterns like model_epoch_10.h5, model.10.h5, ckpt-10 etc.
                filename = os.path.basename(latest_checkpoint)
                epoch_str = ""
                parts = filename.replace(
                    '.weights.h5', '').replace('.h5', '').split('_')
                numeric_parts = [p for p in parts if p.isdigit()]
                if numeric_parts:
                    # Take the last number found after splitting by '_'
                    epoch_str = numeric_parts[-1]
                else:  # Try splitting by '.' or '-'
                    parts = filename.replace('.weights.h5', '').replace(
                        '.h5', '').replace('-', '.').split('.')
                    numeric_parts = [p for p in parts if p.isdigit()]
                    if numeric_parts:
                        epoch_str = numeric_parts[-1]  # Take the last number

                if epoch_str.isdigit():
                    initial_epoch = int(epoch_str)
                    print(
                        f"Successfully loaded weights. Starting from epoch {initial_epoch + 1}")
                else:
                    print(
                        "Warning: Could not determine epoch from checkpoint name. Starting from epoch 0.")
                    initial_epoch = 0
            except Exception as parse_err:
                print(
                    f"Warning: Error parsing epoch from checkpoint name '{latest_checkpoint}': {parse_err}. Starting from epoch 0.")
                initial_epoch = 0
        except Exception as load_err:
            print(f"Error loading weights: {load_err}. Starting from scratch.")
            initial_epoch = 0
    else:
        print("No checkpoint found, starting training from scratch.")

    # Define Callbacks
    # ... existing code for callbacks ...
    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr_step_decay(epoch, lr, initial_lr=initial_learning_rate,
                                        drop=lr_scheduler_drop, epochs_drop=lr_scheduler_epochs_drop),
        verbose=0
    )

    log_dir = os.path.join(
        output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)  # Ensure log dir exists

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
        lr_schedule_callback,
        # Log histograms every epoch
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ConciseProgressCallback(log_frequency=1),  # Keep only one instance
    ]

    # 4. Train the Model (use the mapped dataset)
    epochs_to_run = num_epochs - initial_epoch
    print(
        f"Starting training loop from epoch {initial_epoch + 1} for {epochs_to_run} epochs (Total planned: {num_epochs})...")
    history = None
    if epochs_to_run <= 0:
        print("Training already completed based on initial_epoch. Skipping fit.")
        # Try to load history if it exists, useful if only evaluation is needed later
        hist_csv_file = os.path.join(output_dir, 'training_history.csv')
        if os.path.exists(hist_csv_file):
            print(f"Loading existing training history from {hist_csv_file}")
            try:
                history_df = pd.read_csv(hist_csv_file)
                # Convert DataFrame back to a history-like dictionary object for plotting function
                history = tf.keras.callbacks.History()
                history.history = history_df.to_dict(orient='list')
                history.epoch = list(range(len(history_df)))
            except Exception as e:
                print(f"Warning: Could not load or parse history file: {e}")
                history = None  # Indicate no history available
        else:
            history = None  # Indicate no history available
    else:
        history = model.fit(
            train_dataset_for_fit,       # Use the mapped dataset
            validation_data=val_dataset_for_fit,  # Use the mapped dataset
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=0  # Use custom callback for logging
        )

    # 5. Save Training History
    # ... existing code for saving history ...
    if history and hasattr(history, 'history') and history.history:
        try:
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = os.path.join(output_dir, 'training_history.csv')
            hist_df.to_csv(hist_csv_file, index=False)
            print(f"Training history saved to {hist_csv_file}")
            # Pass variant name for plot title
            plot_training_history(history, output_dir, variant_name)
        except Exception as e:
            print(f"Error saving/plotting training history: {e}")
    # ... rest of existing history saving code ...

    # Cleanup
    print(f"Cleaning up resources after training ({variant_name})...")
    del full_train_dataset, full_val_dataset
    del train_dataset_for_fit, val_dataset_for_fit
    gc.collect()
    print("Cleaned up datasets.")

    return history, model


def evaluate_model(
    model_func,
    dataset_func,
    strategy,
    global_batch_size,
    # --- Constants to pass ---
    variant_name,
    checkpoint_best_path,
    checkpoint_path,
    output_folder,
    img_height,
    img_width,
    input_channels,
    threshold,
    loss_config,
    metrics_list,
    metadata_file,
    h5_data_dir,
    modality_indices,
    rgb_mapping_indices,
    validation_split=0.2,
    num_examples_to_save=5,
    reference_modality_index=3,  # Pass ref modality index to data func
    # --- Optional arguments ---
    model_instance=None,
    **model_kwargs
):
    """Evaluates the trained AS-Net model variant."""
    print(f"\n--- Starting Model Evaluation ({variant_name}) ---")
    evaluation_results = None

    try:
        # 1. Load FULL Validation Data (including display components)
        print(
            f"Loading FULL validation data with global batch size: {global_batch_size}...")
        _, full_val_dataset = dataset_func(
            metadata_file=metadata_file,
            h5_dir=h5_data_dir,
            target_size=(img_height, img_width),
            batch_size=global_batch_size,
            buffer_size=10,
            modality_indices=modality_indices,
            rgb_mapping_indices=rgb_mapping_indices,
            input_channels=input_channels,
            validation_split=validation_split,
            reference_modality_index=reference_modality_index  # Pass argument
        )
        print("Full validation dataset loaded.")

        # Create dataset suitable for model.evaluate (input, binary_mask only)
        val_dataset_for_eval = full_val_dataset.map(
            lambda img_in, mask_bin, img_ref, mask_multi: (img_in, mask_bin))
        print("Created dataset for model.evaluate.")

        # 2. Load or Use Provided Model
        # ... existing code for model loading ...
        model_eval = None
        if model_instance is None:
            print("Loading model weights for evaluation.")
            # ... rest of existing model loading code ...
            checkpoint_to_load = None
            # Prefer best checkpoint
            if os.path.exists(checkpoint_best_path) and os.path.exists(checkpoint_best_path + ".index"):
                print(f"Using best checkpoint: {checkpoint_best_path}")
                checkpoint_to_load = checkpoint_best_path
            else:
                print(
                    f"Warning: Best checkpoint not found or index missing at {checkpoint_best_path}.")
                # Fallback to last epoch checkpoint
                last_checkpoint = tf.train.latest_checkpoint(
                    os.path.dirname(checkpoint_path))
                if last_checkpoint and os.path.exists(last_checkpoint + ".index"):
                    print(
                        f"Attempting to load last epoch checkpoint: {last_checkpoint}")
                    checkpoint_to_load = last_checkpoint
                else:
                    print(
                        f"Error: No suitable checkpoint found in {os.path.dirname(checkpoint_path)}. Cannot evaluate.")
                    return None

            print("Rebuilding model architecture for evaluation...")
            with strategy.scope():
                model_eval = model_func(
                    input_size=(img_height, img_width, input_channels),
                    **model_kwargs  # Pass variant etc. if needed
                )
                print("Compiling evaluation model...")
                loss_instance = CombinedLoss(**loss_config)
                # Optimizer state not loaded/used but needed for compile
                optimizer = tf.keras.optimizers.Adam()
                if mixed_precision.global_policy().name == 'mixed_float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                # Ensure metrics are instantiated correctly for compilation
                compiled_metrics = []
                for m in metrics_list:
                    if isinstance(m, str):
                        compiled_metrics.append(m)
                    elif hasattr(m, 'get_config'):
                        # Fix here: Handle Precision and Recall metrics specifically
                        if m.__class__.__name__ == 'Precision' or m.__class__.__name__ == 'Recall':
                            config = m.get_config()
                            if 'threshold' in config:
                                # Convert singular 'threshold' to plural 'thresholds'
                                threshold_value = config.pop('threshold')
                                config['thresholds'] = threshold_value
                            compiled_metrics.append(
                                m.__class__.from_config(config))
                        else:
                            config = m.get_config()
                            if 'threshold' in config:
                                config['threshold'] = threshold
                            compiled_metrics.append(
                                m.__class__.from_config(config))
                    else:
                        compiled_metrics.append(m)

                model_eval.compile(optimizer=optimizer,
                                   loss=loss_instance, metrics=compiled_metrics)
                print(f"Loading weights from {checkpoint_to_load}...")
                load_status = model_eval.load_weights(checkpoint_to_load)
                # Allow optimizer state mismatch etc.
                load_status.expect_partial()
                print("Successfully loaded weights into new model instance.")
        else:
            model_eval = model_instance
            # ... rest of existing model_instance handling ...
            # Re-compile if necessary (e.g., if metrics weren't compiled correctly before)
            if not model_eval.compiled_metrics or not model_eval.optimizer:
                print("Compiling the provided model instance for evaluation...")
                with strategy.scope():
                    loss_instance = CombinedLoss(**loss_config)
                    optimizer = tf.keras.optimizers.Adam()
                    if mixed_precision.global_policy().name == 'mixed_float16':
                        optimizer = mixed_precision.LossScaleOptimizer(
                            optimizer)

                    compiled_metrics = []
                    for m in metrics_list:
                        if isinstance(m, str):
                            compiled_metrics.append(m)
                        elif hasattr(m, 'get_config'):
                            # Fix here: Handle Precision and Recall metrics specifically
                            if m.__class__.__name__ == 'Precision' or m.__class__.__name__ == 'Recall':
                                config = m.get_config()
                                if 'threshold' in config:
                                    # Convert singular 'threshold' to plural 'thresholds'
                                    threshold_value = config.pop('threshold')
                                    config['thresholds'] = threshold_value
                                compiled_metrics.append(
                                    m.__class__.from_config(config))
                            else:
                                config = m.get_config()
                                if 'threshold' in config:
                                    config['threshold'] = threshold
                                compiled_metrics.append(
                                    m.__class__.from_config(config))
                        else:
                            compiled_metrics.append(m)

                    model_eval.compile(
                        optimizer=optimizer, loss=loss_instance, metrics=compiled_metrics)
                    print("Provided model compiled.")

        # 3. Evaluate (Use the mapped dataset)
        print("Evaluating model on validation set...")
        evaluation_results = model_eval.evaluate(
            val_dataset_for_eval,  # Use the mapped dataset
            verbose=1,
            return_dict=True
        )
        # ... existing code for printing metrics, calculating F1 ...
        print("\nKeras Evaluation Results:")
        metric_order = ['loss', 'binary_accuracy', 'dice_coef',
                        'iou', 'precision', 'recall']  # Preferred order
        # Print in order, then others
        for name in metric_order:
            if name in evaluation_results:
                print(f"- {name}: {evaluation_results[name]:.4f}")
        for name, value in evaluation_results.items():
            if name not in metric_order:
                print(f"- {name}: {value:.4f}")

        # 4. Calculate F1 Score
        precision_val = evaluation_results.get('precision', 0.0)
        recall_val = evaluation_results.get('recall', 0.0)
        f1_val = 0.0
        if (precision_val + recall_val) > 1e-7:
            f1_val = 2 * (precision_val * recall_val) / \
                (precision_val + recall_val)
        evaluation_results['f1_score'] = f1_val  # Add to dict
        print(f"- f1_score: {f1_val:.4f} (calculated)")

        # 5. Save Performance Metrics
        # ... existing code ...
        try:
            perf_file_path = os.path.join(output_folder, "performances.txt")
            with open(perf_file_path, "w") as file_perf:
                file_perf.write(f"Evaluation Metrics ({variant_name}):\n")
                file_perf.write("------------------------------------\n")
                metric_order.append('f1_score')  # Add f1 to preferred order
                for name in metric_order:
                    if name in evaluation_results:
                        file_perf.write(
                            f"- {name.replace('_', ' ').title()}: {evaluation_results[name]:.4f}\n")
                for name, value in evaluation_results.items():  # Add any others
                    if name not in metric_order:
                        file_perf.write(
                            f"- {name.replace('_', ' ').title()}: {value:.4f}\n")

            print(f"Evaluation results saved to {perf_file_path}")
        except Exception as e:
            print(f"Error saving performance metrics: {e}")

        # 6. Save Prediction Examples (Use the FULL dataset)
        print("\nGenerating prediction examples...")
        # Pass the FULL validation dataset to the updated save function
        save_prediction_examples(
            model=model_eval,
            dataset=full_val_dataset,  # Pass the one with all components
            output_folder=output_folder,
            num_examples=num_examples_to_save,
            threshold=threshold
        )

        print(f"--- Evaluation Finished ({variant_name}) ---")
        return evaluation_results

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        print("Cleaning up resources after evaluation...")
        if 'full_val_dataset' in locals():
            del full_val_dataset
        if 'val_dataset_for_eval' in locals():
            del val_dataset_for_eval
        if 'model_eval' in locals() and model_eval is not model_instance:
            del model_eval
        gc.collect()
        print("Cleaned up evaluation resources.")


def save_prediction_examples(model, dataset, output_folder, num_examples=5, threshold=0.5):
    """
    Saves example predictions comparing reference image, ground truth, and prediction overlay.
    Assumes dataset yields: (image_model_input, mask_binary_gt, image_display_ref, mask_multiclass_gt)
    """
    print(f"Saving {num_examples} comparison prediction examples...")
    examples_dir = os.path.join(output_folder, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Define colors for segmentation maps (Consistent with explore_dataset.py)
    # Background=Transparent, NCR/NET=Blue, ED=Green, ET=Yellow
    # 0=TransparentBG, 1=NCR/NET, 2=ED, 3=Skip(Black), 4=ET (BraTS values are 1, 2, 4)
    brats_colors = ['#FFFFFF00', '#3b528b', '#18b880', '#000000', '#e6d74f']
    brats_cmap = mcolors.ListedColormap(brats_colors)
    # Boundaries for values 0, 1, 2, 4. Need boundaries for these.
    # -0.5 | 0.5 | 1.5 | 2.5 | 3.5 is skipped | 4.5
    brats_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 4.5], brats_cmap.N)

    # Define color for binary prediction overlay
    prediction_color = 'red'  # Color for the model's prediction mask

    try:
        # Take one batch from the FULL dataset
        for batch_data in dataset.take(1):
            # Unpack the batch data
            if len(batch_data) != 4:
                print(
                    f"ERROR: Dataset expected 4 components, got {len(batch_data)}. Cannot save examples.")
                print(
                    "Expected: (image_model_input, mask_binary_gt, image_display_ref, mask_multiclass_gt)")
                return

            image_model_inputs, _, image_display_refs, mask_multiclass_gts = batch_data
            batch_size = image_model_inputs.shape[0]
            print(
                f"Generating predictions for {min(num_examples, batch_size)} examples...")

            # Predict using the model input part of the batch
            predictions = model.predict(image_model_inputs)
            # Apply threshold for binary prediction map
            binary_predictions = tf.cast(
                predictions >= threshold, tf.float32).numpy()  # (B, H, W, 1)

            # (B, H, W) - Already normalized 0-1
            image_display_refs = image_display_refs.numpy()
            # (B, H, W) - Values 0, 1, 2, 4
            mask_multiclass_gts = mask_multiclass_gts.numpy()

            print("Plotting and saving examples...")
            for j in range(min(num_examples, batch_size)):
                # --- Create 3 side-by-side plots ---
                # 3 plots, adjust size
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                ref_img = image_display_refs[j]         # (H, W)
                gt_mask_multi = mask_multiclass_gts[j]  # (H, W)
                pred_binary = binary_predictions[j].squeeze()  # (H, W)

                # --- Plot 1: Reference Modality (e.g., FLAIR) ---
                axes[0].imshow(ref_img, cmap='gray')
                axes[0].set_title("Reference Image (FLAIR)")
                axes[0].axis("off")

                # --- Plot 2: Ground Truth Overlay ---
                axes[1].imshow(ref_img, cmap='gray', alpha=0.8)
                # Overlay GT mask, making background (0) transparent
                gt_viz = axes[1].imshow(gt_mask_multi, cmap=brats_cmap, norm=brats_norm,
                                        alpha=0.6 * (gt_mask_multi > 0))  # Slightly stronger alpha
                axes[1].set_title("Ground Truth Overlay")
                axes[1].axis("off")
                # Create legend for GT
                patches_gt = [mpatches.Patch(color=brats_cmap(brats_norm(val)), label=label)
                              for val, label in zip([1, 2, 4], ['NCR/NET', 'ED', 'ET'])]
                axes[1].legend(handles=patches_gt, loc='lower right',
                               fontsize='small', title="GT Regions")

                # --- Plot 3: Model Prediction Overlay ---
                axes[2].imshow(ref_img, cmap='gray', alpha=0.8)
                # Create a colored map just for the prediction foreground
                pred_overlay = np.zeros((*pred_binary.shape, 4))  # RGBA
                pred_overlay[pred_binary > 0] = mcolors.to_rgba(
                    prediction_color, alpha=0.7)  # Use desired color, stronger alpha
                axes[2].imshow(pred_overlay)
                axes[2].set_title(
                    f"Model Prediction (Threshold={threshold:.2f})")
                axes[2].axis("off")
                # Add simple legend for prediction color
                pred_patch = mpatches.Patch(
                    color=prediction_color, label='Predicted Tumor')
                axes[2].legend(handles=[pred_patch],
                               loc='lower right', fontsize='small')

                plt.suptitle(f"Prediction Example {j+1}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

                example_save_path = os.path.join(
                    examples_dir, f"comparison_example_{j+1}.png")
                plt.savefig(example_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)  # Close the figure

            print(f"Saved comparison prediction examples to {examples_dir}")
            break  # Only process one batch

    except Exception as e:
        print(f"Error saving comparison prediction examples: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()

# --- Training Callbacks ---


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
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k,
                                     v in logs.items()])
            print(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - {metrics_str}")
            try:  # Monitor Synergy weights if layer exists
                synergy_layer = self.model.get_layer('synergy')
                alpha_val = synergy_layer.alpha.numpy()
                beta_val = synergy_layer.beta.numpy()
                print(
                    f"    Synergy weights - alpha: {alpha_val:.4f}, beta: {beta_val:.4f}")
            except Exception:
                pass  # Layer might not exist
        gc.collect()  # Force garbage collection

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n--- Training Finished ---")
        print(f"Total training time: {total_time:.2f} seconds")


def lr_step_decay(epoch, lr, initial_lr, drop=0.5, epochs_drop=10):
    """Applies step decay to the learning rate. Needs initial_lr passed."""
    new_lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    final_lr = max(new_lr, 1e-7)  # Minimum LR
    return final_lr

# --- Plot Training History ---


def plot_training_history(history, output_dir, variant_name="Model"):
    """Plots training & validation loss and metrics and saves the plot."""
    print("--- Plotting Training History ---")
    try:
        history_dict = history.history
        if not history_dict:
            print("History object is empty. Skipping plotting.")
            return

        epochs = range(1, len(history_dict['loss']) + 1)
        metrics_to_plot = {'loss': 'Loss'}
        # Dynamically find metrics present in history
        for key in history_dict.keys():
            if key.startswith('val_') or key == 'lr':
                continue  # Skip val metrics and learning rate here
            # Only plot if validation counterpart exists
            if key != 'loss' and f'val_{key}' in history_dict:
                # Create a user-friendly title
                title = key.replace('_', ' ').title()
                if key == 'dice_coef':
                    title = 'Dice Coefficient'
                elif key == 'iou':
                    title = 'IoU'
                elif key == 'binary_accuracy':
                    title = 'Accuracy'
                metrics_to_plot[key] = title

        num_plots = len(metrics_to_plot)
        if num_plots <= 0:
            print("Warning: No suitable metrics found to plot (excluding loss).")
            return

        plt.figure(figsize=(max(12, 6 * num_plots), 5))

        plot_index = 1
        for metric, title in metrics_to_plot.items():
            plt.subplot(1, num_plots, plot_index)
            val_metric = f'val_{metric}'
            if metric in history_dict:
                plt.plot(epochs, history_dict[metric],
                         'bo-', label=f'Training {title}')
            if val_metric in history_dict:
                plt.plot(epochs, history_dict[val_metric],
                         'ro-', label=f'Validation {title}')
            else:  # Only training data available for this metric
                if metric in history_dict:
                    plt.legend()

            plt.title(f'{title}')
            plt.xlabel('Epoch')
            # Only add legend if both training and validation data are plotted
            if metric in history_dict and val_metric in history_dict:
                plt.legend()
            if metric != 'loss':  # Set y-axis limits for common metrics
                # Check min/max to set appropriate limits if they exceed [0,1]
                min_val = min(history_dict.get(
                    metric, [0]) + history_dict.get(val_metric, [0]))
                max_val = max(history_dict.get(
                    metric, [1]) + history_dict.get(val_metric, [1]))
                if min_val >= 0 and max_val <= 1.1:  # Allow a bit of margin above 1
                    plt.ylim([0, 1.05])
                # Else: use default ylim scaling
            plt.grid(True)
            plot_index += 1

        plt.suptitle(f'{variant_name} Training History', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
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


# --- Completion Notification ---


def create_completion_notification(
    variant_name,
    output_folder,
    completion_file,
    # Constants to include
    img_height,
    img_width,
    input_channels,
    batch_size_per_replica,
    global_batch_size,
    num_epochs,
    initial_learning_rate,
    loss_config,
    checkpoint_dir,
    checkpoint_best_path,
    # Optional timing
    start_time=None
):
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
            f.write(
                f"AS-Net ({variant_name}) Training Completed at: {timestamp}\n\n")
            f.write("Training Configuration:\n")
            f.write(f"- Model Variant: {variant_name}\n")
            f.write(f"- Image dimensions: {img_height}x{img_width}\n")
            f.write(f"- Input Channels: {input_channels}\n")
            f.write(f"- Batch size (per replica): {batch_size_per_replica}\n")
            f.write(f"- Global Batch size: {global_batch_size}\n")
            f.write(f"- Epochs planned: {num_epochs}\n")
            f.write(f"- Initial Learning rate: {initial_learning_rate}\n")
            f.write(
                f"- Mixed Precision Policy: {mixed_precision.global_policy().name}\n")
            f.write(f"- Loss Config: {loss_config}\n")
            f.write(f"- Total Duration: {duration_str}\n\n")

            f.write("Checkpoint and output locations:\n")
            f.write(f"- Checkpoint directory: {checkpoint_dir}\n")
            f.write(f"- Best model weights: {checkpoint_best_path}\n")
            f.write(f"- Output directory: {output_folder}\n")

            f.write("\n--- Final Performance Metrics ---\n")
            if os.path.exists(perf_file_path):
                try:
                    with open(perf_file_path, "r") as perf_file:
                        f.write(perf_file.read())
                except Exception as read_err:
                    f.write(
                        f"Note: Error reading performance file ({perf_file_path}): {read_err}\n")
            else:
                f.write(
                    f"Note: Performance file not found ({perf_file_path}). Evaluation failed or not run.\n")

        print(f"Completion notification saved to: {completion_file}")

    except Exception as e:
        print(f"Error creating completion notification file: {e}")


# --- Final Script Cleanup Utility ---
def final_cleanup(model=None, history=None, evaluation_results=None):
    print("\n--- Final Script Cleanup ---")
    # Clear potentially large objects
    if model is not None:
        del model
    if history is not None:
        del history
    if evaluation_results is not None:
        del evaluation_results
    # Datasets are usually deleted within train/eval functions
    gc.collect()
    backend.clear_session()
    print("Script execution completed.")
