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
    MaxPooling2D,
    UpSampling2D,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Add,
    Layer,
)
from keras.losses import Loss
from keras.metrics import Metric
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tensorflow.keras import mixed_precision
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from tensorflow.keras.callbacks import TensorBoard
import datetime
import glob


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

    # Optional JIT configuration
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
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3])


# --- AS-Net Core Modules (SAM, CAM, Synergy) ---

class SAM(Model):
    """Spatial Attention Module"""

    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_channels = max(
            16, filters // 8 if filters > 128 else filters // 4)
        compute_dtype = mixed_precision.global_policy().compute_dtype

        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv4')

        self.pool1 = MaxPooling2D((2, 2), name='pool1')
        self.upsample1 = UpSampling2D(
            (2, 2), interpolation="bilinear", name='upsample1')
        self.W1 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal",
                         dtype=compute_dtype, name='W1_map')

        self.pool2 = MaxPooling2D((4, 4), name='pool2')
        self.upsample2 = UpSampling2D(
            (4, 4), interpolation="bilinear", name='upsample2')
        self.W2 = Conv2D(1, 1, activation="sigmoid", kernel_initializer="he_normal",
                         dtype=compute_dtype, name='W2_map')

        self.add_attention = Add(name='add_attention')
        self.multiply_attention = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')
        self.resize_layer = ResizeToMatchLayer(name=f'{name}_resize_attention')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))  # F'(X)
        out2 = self.conv4(inputs)  # F''(X)

        pool1 = self.pool1(out2)
        up1 = self.upsample1(pool1)
        up1 = self.resize_layer([up1, out2])
        att1 = self.W1(up1)  # S1

        pool2 = self.pool2(out2)
        up2 = self.upsample2(pool2)
        up2 = self.resize_layer([up2, out2])
        att2 = self.W2(up2)  # S2

        attention_map = self.add_attention([att1, att2])  # S

        attended_features = self.multiply_attention([out1, attention_map])
        y = self.add_residual([attended_features, out2]
                              )  # Y = F'(X)*S + F''(X)
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
        self.out_channels = max(
            16, filters // 8 if filters > 128 else filters // 4)
        self.reduction_ratio = reduction_ratio
        compute_dtype = mixed_precision.global_policy().compute_dtype

        self.conv1 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv1')
        self.conv2 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv2')
        self.conv3 = Conv2D(self.out_channels, 3, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv3')
        self.conv4 = Conv2D(self.out_channels, 1, activation="relu", padding="same",
                            kernel_initializer="he_normal", dtype=compute_dtype, name='conv4')

        self.gpool = GlobalAveragePooling2D(
            name='global_avg_pool', keepdims=True)
        reduced_channels = max(1, self.out_channels // self.reduction_ratio)
        self.fc1 = Dense(reduced_channels, activation="relu",
                         use_bias=False, dtype=compute_dtype, name='fc1')
        self.fc2 = Dense(self.out_channels, activation="sigmoid",
                         use_bias=False, dtype=compute_dtype, name='fc2')

        self.multiply_attention = Multiply(name='multiply_attention')
        self.add_residual = Add(name='add_residual')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))  # F'(X)
        out2 = self.conv4(inputs)  # F''(X)

        pooled = self.gpool(out2)
        fc1_out = self.fc1(pooled)
        channel_attention_weights = self.fc2(fc1_out)

        recalibrated_features = self.multiply_attention(
            [out1, channel_attention_weights])
        # Y = F'(X)*C + F''(X)
        y = self.add_residual([recalibrated_features, out2])
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
        self.alpha = tf.Variable(
            alpha_init, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(beta_init, trainable=True,
                                name="beta", dtype=tf.float32)
        compute_dtype = mixed_precision.global_policy().compute_dtype

        self.conv = Conv2D(
            1, 1, padding="same", kernel_initializer="he_normal", dtype=compute_dtype, name='conv')
        self.bn = BatchNormalization(name='bn')
        self.add = Add(name='add_weighted')

    def call(self, inputs):
        sam_features, cam_features = inputs
        compute_dtype = sam_features.dtype
        alpha_casted = tf.cast(self.alpha, compute_dtype)
        beta_casted = tf.cast(self.beta, compute_dtype)

        weighted_sum = self.add(
            [alpha_casted * sam_features, beta_casted * cam_features])
        convolved = self.conv(weighted_sum)
        bn_out = self.bn(convolved)
        return bn_out

    def get_config(self):
        config = super(Synergy, self).get_config()
        config.update({"alpha_init": 0.5, "beta_init": 0.5})
        return config


# --- Loss Functions ---

class DiceLoss(Loss):
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
    def __init__(self, weight=1.0, name='weighted_bce_loss', **kwargs):
        super(WBCE, self).__init__(
            name=name, reduction='sum_over_batch_size', **kwargs)
        self.weight = tf.cast(weight, tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon_ = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
        logits = tf.math.log(y_pred / (1.0 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=tf.cast(self.weight, logits.dtype)
        )
        return loss

    def get_config(self):
        config = super(WBCE, self).get_config()
        config.update({"weight": self.weight.numpy()})
        return config


class CombinedLoss(Loss):
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


# --- Data Preparation (MODIFIED for BraTS 2023 H5 Slices) ---

def prepare_brats_data_gpu(
    h5_base_dir,        # Base directory containing train/validation/test subdirs
    split_type,         # 'train', 'validation', or 'test'
    target_size,        # Tuple (height, width) for resizing
    batch_size,         # Global batch size
    buffer_size,        # Shuffle buffer size (only for training)
    input_channels=3,   # Should match preprocessed H5 image channels
):
    """
    Prepares the BraTS 2023 dataset from preprocessed H5 slices using tf.data.
    Assumes H5 files contain 'image' (H,W,C, float32, pre-normalized, RGB-mapped)
    and 'mask' (H,W,1, float32, binary).
    Outputs: (image_model_input, mask_binary_gt)
    """
    print(f"--- Setting up Data Pipeline (BraTS 2023 H5 - {split_type}) ---")
    print(f"H5 Base Directory: {h5_base_dir}")
    print(f"Target image size: {target_size}")

    # Determine the directory for the specified split
    h5_split_dir = os.path.join(h5_base_dir, split_type)
    if not os.path.isdir(h5_split_dir):
        raise FileNotFoundError(
            f"H5 data directory for split '{split_type}' not found at {h5_split_dir}")

    # List all H5 files in the split directory
    h5_files = glob.glob(os.path.join(h5_split_dir, "*.h5"))
    if not h5_files:
        raise ValueError(f"No H5 files found in {h5_split_dir}.")
    print(f"Found {len(h5_files)} H5 slice files for the '{split_type}' split.")

    # --- Parse function for the new H5 structure ---
    def parse_preprocessed_h5_file(file_path):
        def _parse_h5(path_tensor):
            path = path_tensor.numpy().decode("utf-8")
            try:
                with h5py.File(path, "r") as hf:
                    # Load preprocessed image (already normalized, 3 channels)
                    image_preprocessed = hf["image"][()].astype(
                        np.float32)  # (H, W, 3)
                    # Load binary mask (already 0/1)
                    mask_binary = hf["mask"][()].astype(
                        np.float32)         # (H, W, 1)

                    # Assume original size is needed to set shape initially
                    original_h, original_w = image_preprocessed.shape[:2]

                    return image_preprocessed, mask_binary, original_h, original_w

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                # Return dummy data of expected type and rough shape if loading fails
                # Using 240x240 as a common BraTS size
                dummy_h, dummy_w = 240, 240
                dummy_image = np.zeros(
                    (dummy_h, dummy_w, input_channels), dtype=np.float32)
                dummy_mask = np.zeros((dummy_h, dummy_w, 1), dtype=np.float32)
                return dummy_image, dummy_mask, dummy_h, dummy_w

        # Update output types
        image_data, mask_data, h, w = tf.py_function(
            _parse_h5, [file_path], [tf.float32,
                                     tf.float32, tf.int32, tf.int32]
        )

        # Set shapes based on loaded dimensions (dynamic per slice)
        # Note: Setting dynamic shapes like this might sometimes cause issues.
        # If problems arise, consider finding a fixed original shape (e.g., 240x240)
        # if all slices conform to it after NIfTI loading.
        # image_data.set_shape([h, w, input_channels]) # --> Leads to tf errors usually
        # mask_data.set_shape([h, w, 1])              # --> Leads to tf errors usually
        # Let's try setting a known fixed shape (common BraTS size) - adjust if different!
        # This assumes the preprocessing script produces consistent sizes, e.g., 240x240
        assumed_h, assumed_w = 240, 240
        image_data.set_shape([assumed_h, assumed_w, input_channels])
        mask_data.set_shape([assumed_h, assumed_w, 1])

        return image_data, mask_data

    # --- Preprocess function (mainly resizing) ---
    def resize_data(image_data, mask_data):
        # Resize image
        image_resized = tf.image.resize(
            image_data, target_size, method='bilinear')
        image_final = tf.cast(image_resized, tf.float32)  # Ensure float32

        # Resize mask using nearest neighbor to keep binary values crisp
        mask_resized = tf.image.resize(
            mask_data, target_size, method='nearest')
        # Ensure mask is binary 0/1 after resize
        mask_final = tf.cast(mask_resized > 0.5, tf.float32)

        # Set Final Shapes after resizing
        image_final.set_shape([target_size[0], target_size[1], input_channels])
        mask_final.set_shape([target_size[0], target_size[1], 1])

        return image_final, mask_final

    # --- Create tf.data Datasets ---
    dataset = tf.data.Dataset.from_tensor_slices(h5_files)

    # --- Apply Transformations ---
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    # Shuffle only for training set
    if split_type == 'train':
        dataset = dataset.shuffle(buffer_size)

    dataset = (
        dataset
        .map(parse_preprocessed_h5_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(resize_data, num_parallel_calls=tf.data.AUTOTUNE)  # Apply resizing
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Data pipeline created successfully for {split_type} split.")
    print("Dataset Element Spec:", dataset.element_spec)
    # Expected Output: ( TensorSpec(shape=(None, TargetH, TargetW, 3), dtype=tf.float32, name=None), <- Model Input
    #                     TensorSpec(shape=(None, TargetH, TargetW, 1), dtype=tf.float32, name=None) ) <- Binary Mask (Loss)

    return dataset


# --- Training Function ---
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
    h5_data_dir,
    buffer_size,
    lr_scheduler_drop=0.5,
    lr_scheduler_epochs_drop=10,
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

    # 1. Create Datasets using the modified loader
    print("Preparing datasets...")
    train_dataset = dataset_func(
        h5_base_dir=h5_data_dir,
        split_type='train',
        target_size=(img_height, img_width),
        batch_size=global_batch_size,
        buffer_size=buffer_size,
        input_channels=input_channels,
    )
    val_dataset = dataset_func(
        h5_base_dir=h5_data_dir,
        split_type='validation',
        target_size=(img_height, img_width),
        batch_size=global_batch_size,
        buffer_size=buffer_size,  # Not used for validation shuffle, but required arg
        input_channels=input_channels,
    )
    print("Train and Validation datasets prepared.")
    print("Train Dataset Spec:", train_dataset.element_spec)

    # 2. Build and Compile Model (within strategy scope)
    with strategy.scope():
        print("Building model...")
        # Ensure model_func gets necessary kwargs (like 'variant' for MobileNet/EfficientNet)
        model = model_func(
            input_size=(img_height, img_width, input_channels),
            **model_kwargs
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
                config = m.get_config()
                # Handle Keras Precision/Recall specifically expecting 'thresholds' plural
                if m.__class__.__name__ in ['Precision', 'Recall']:
                    if 'thresholds' not in config and 'threshold' in config:
                        config['thresholds'] = config.pop(
                            'threshold')
                    elif 'thresholds' not in config:  # If neither exists, add default
                        config['thresholds'] = threshold
                    # Ensure threshold is list/tuple if present
                    if 'thresholds' in config and not isinstance(config['thresholds'], (list, tuple)):
                        config['thresholds'] = [config['thresholds']]
                elif 'threshold' in config:  # For custom Dice/IoU
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

    # 3. Check for Checkpoint Resume
    latest_checkpoint = tf.train.latest_checkpoint(
        os.path.dirname(checkpoint_path))
    initial_epoch = 0
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        try:
            with strategy.scope():
                # Use expect_partial for flexibility if optimizer state isn't saved/needed
                model.load_weights(latest_checkpoint).expect_partial()

            # Simple epoch extraction (adjust if checkpoint naming differs significantly)
            try:
                filename = os.path.basename(latest_checkpoint)
                epoch_str = ""
                # Try common patterns like model_epoch_10.h5, model.10.h5, ckpt-10 etc.
                parts = filename.replace('.weights.h5', '').replace(
                    '.h5', '').split('_')[-1].split('.')[-1].split('-')[-1]
                if parts.isdigit():
                    epoch_str = parts
                else:  # Fallback if no number found at the end
                    numeric_parts = [p for p in filename.replace(
                        '.weights.h5', '').replace('.h5', '').split('_') if p.isdigit()]
                    if numeric_parts:
                        epoch_str = numeric_parts[-1]

                if epoch_str.isdigit():
                    # Assume epoch number is 0-based in filename
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

    # 4. Define Callbacks
    # Custom LR scheduler needs access to initial_learning_rate

    def lr_schedule_wrapper(epoch, lr):
        return lr_step_decay(epoch, lr, initial_lr=initial_learning_rate, drop=lr_scheduler_drop, epochs_drop=lr_scheduler_epochs_drop)

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule_wrapper, verbose=0)

    log_dir = os.path.join(
        output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

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

    # 5. Train the Model
    epochs_to_run = num_epochs - initial_epoch
    print(
        f"Starting training loop from epoch {initial_epoch + 1} for {epochs_to_run} epochs (Total planned: {num_epochs})...")
    history = None
    if epochs_to_run <= 0:
        print("Training already completed based on initial_epoch. Skipping fit.")
        # Try to load history if it exists
        hist_csv_file = os.path.join(output_dir, 'training_history.csv')
        if os.path.exists(hist_csv_file):
            print(f"Loading existing training history from {hist_csv_file}")
            try:
                history_df = pd.read_csv(hist_csv_file)
                history = tf.keras.callbacks.History()
                history.history = history_df.to_dict(orient='list')
                history.epoch = list(range(len(history_df)))
            except Exception as e:
                print(f"Warning: Could not load or parse history file: {e}")
                history = None
        else:
            history = None  # Indicate no history available
    else:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=0  # Use custom callback for logging
        )

    # 6. Save Training History
    if history and hasattr(history, 'history') and history.history:
        try:
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = os.path.join(output_dir, 'training_history.csv')
            hist_df.to_csv(hist_csv_file, index=False)
            print(f"Training history saved to {hist_csv_file}")
            plot_training_history(history, output_dir, variant_name)
        except Exception as e:
            print(f"Error saving/plotting training history: {e}")

    # 7. Cleanup
    print(f"Cleaning up resources after training ({variant_name})...")
    del train_dataset, val_dataset
    gc.collect()
    print("Cleaned up datasets.")

    return history, model


# --- Evaluation Function (MODIFIED for test split and new data loader) ---
def evaluate_model(
    model_func,
    dataset_func,
    strategy,
    global_batch_size,
    # --- Constants to pass ---
    variant_name,
    checkpoint_best_path,  # Path to best weights
    checkpoint_path,      # Path for latest weights (fallback)
    output_folder,        # For saving results/examples
    img_height,
    img_width,
    input_channels,       # Keep this
    threshold,            # For metrics and prediction examples
    loss_config,          # Dictionary for CombinedLoss
    metrics_list,         # List of metric objects/names
    h5_data_dir,          # Base H5 directory
    # split_dir,          # Directory with train/val/test ID files (Not needed if data_func lists files)
    num_examples_to_save=5,
    # --- Optional arguments ---
    model_instance=None,  # Pass trained model if available
    **model_kwargs        # Pass args like 'variant' to model_func
):
    """Evaluates the trained AS-Net model variant on the TEST split."""
    print(f"\n--- Starting Model Evaluation on TEST data ({variant_name}) ---")
    evaluation_results = None
    inference_timing = {}

    try:
        # 1. Load TEST Data using the modified loader
        print(
            f"Loading TEST data with global batch size: {global_batch_size}...")
        test_dataset = dataset_func(
            h5_base_dir=h5_data_dir,
            split_type='test',  # <-- Use test split
            target_size=(img_height, img_width),
            batch_size=global_batch_size,
            buffer_size=10,  # Not used for shuffle, placeholder
            input_channels=input_channels,
        )
        # Count total number of test slices for timing calculations
        num_test_slices = 0
        try:
            # Efficiently count elements without loading data
            # Note: This might iterate once if cardinality is unknown
            num_test_slices = test_dataset.reduce(
                0, lambda x, _: x + tf.shape(_[0])[0]).numpy()
            print(f"Found {num_test_slices} total test slices.")
        except Exception as count_err:
            print(
                f"Warning: Could not accurately determine total test slices: {count_err}. Per-slice timing might be inaccurate.")
            num_test_slices = -1  # Indicate unknown count

        print("Test dataset loaded.")

        # 2. Load or Use Provided Model
        model_eval = None
        if model_instance is None:
            print("Loading model weights for evaluation.")
            checkpoint_to_load = None
            # Prefer best checkpoint based on validation performance
            best_ckpt_index = checkpoint_best_path + ".index"  # Check for index file too
            # Check if the .weights.h5 file exists (or the base name if extension is different)
            best_ckpt_base = os.path.splitext(checkpoint_best_path)[0]
            # Find files starting with the base name
            best_ckpt_files = glob.glob(best_ckpt_base + "*")
            # Check for sharded data files
            best_ckpt_data_exists = any('.data-' in f for f in best_ckpt_files)

            if best_ckpt_files and (os.path.exists(best_ckpt_index) or best_ckpt_data_exists):
                print(
                    f"Using best checkpoint (from validation): {checkpoint_best_path}")
                checkpoint_to_load = checkpoint_best_path  # Use the base path for loading
            else:
                print(
                    f"Warning: Best checkpoint files not found or index/data missing near {checkpoint_best_path}.")
                # Fallback to last epoch checkpoint
                last_checkpoint_dir = os.path.dirname(checkpoint_path)
                last_checkpoint = tf.train.latest_checkpoint(
                    last_checkpoint_dir)
                if last_checkpoint:
                    # Check for index/data files for the latest checkpoint
                    last_ckpt_index = last_checkpoint + ".index"
                    last_ckpt_base = os.path.splitext(last_checkpoint)[0]
                    last_ckpt_files = glob.glob(last_ckpt_base + "*")
                    last_ckpt_data_exists = any(
                        '.data-' in f for f in last_ckpt_files)

                    if last_ckpt_files and (os.path.exists(last_ckpt_index) or last_ckpt_data_exists):
                        print(
                            f"Attempting to load last epoch checkpoint: {last_checkpoint}")
                        checkpoint_to_load = last_checkpoint
                    else:
                        print(
                            f"Error: Last checkpoint file or index/data missing ({last_checkpoint}). Cannot evaluate.")
                        return None, None
                else:
                    print(
                        f"Error: No suitable checkpoint found in {last_checkpoint_dir}. Cannot evaluate.")
                    return None, None

            print("Rebuilding model architecture for evaluation...")
            with strategy.scope():
                model_eval = model_func(
                    input_size=(img_height, img_width, input_channels),
                    **model_kwargs  # Pass variant etc. if needed
                )
                print("Compiling evaluation model...")
                loss_instance = CombinedLoss(**loss_config)
                # Optimizer state not strictly needed for eval
                optimizer = tf.keras.optimizers.Adam()
                if mixed_precision.global_policy().name == 'mixed_float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                # Ensure metrics are instantiated correctly for compilation
                compiled_metrics = []
                for m in metrics_list:
                    # (Metric instantiation logic remains the same)
                    if isinstance(m, str):
                        compiled_metrics.append(m)
                    elif hasattr(m, 'get_config'):
                        config = m.get_config()
                        if m.__class__.__name__ in ['Precision', 'Recall']:
                            if 'thresholds' not in config and 'threshold' in config:
                                config['thresholds'] = config.pop('threshold')
                            elif 'thresholds' not in config:
                                config['thresholds'] = threshold
                            if 'thresholds' in config and not isinstance(config['thresholds'], (list, tuple)):
                                config['thresholds'] = [config['thresholds']]
                        elif 'threshold' in config:
                            config['threshold'] = threshold
                        compiled_metrics.append(
                            m.__class__.from_config(config))
                    else:
                        compiled_metrics.append(m)

                model_eval.compile(optimizer=optimizer,
                                   loss=loss_instance, metrics=compiled_metrics)
                print(f"Loading weights from {checkpoint_to_load}...")
                # Use the base path for loading, TF handles sharded files
                load_status = model_eval.load_weights(checkpoint_to_load)
                # Allow optimizer state mismatch etc.
                load_status.expect_partial()
                print("Successfully loaded weights into new model instance.")
        else:
            model_eval = model_instance
            print("Using provided model instance for evaluation.")
            # Re-compile if necessary (e.g., if metrics weren't compiled correctly before)
            if not hasattr(model_eval, 'compiled_metrics') or not model_eval.compiled_metrics or not model_eval.optimizer:
                print("Compiling the provided model instance for evaluation...")
                with strategy.scope():
                    loss_instance = CombinedLoss(**loss_config)
                    optimizer = tf.keras.optimizers.Adam()
                    if mixed_precision.global_policy().name == 'mixed_float16':
                        optimizer = mixed_precision.LossScaleOptimizer(
                            optimizer)

                    compiled_metrics = []
                    for m in metrics_list:
                        # (Metric instantiation logic remains the same)
                        if isinstance(m, str):
                            compiled_metrics.append(m)
                        elif hasattr(m, 'get_config'):
                            config = m.get_config()
                            if m.__class__.__name__ in ['Precision', 'Recall']:
                                if 'thresholds' not in config and 'threshold' in config:
                                    config['thresholds'] = config.pop(
                                        'threshold')
                                elif 'thresholds' not in config:
                                    config['thresholds'] = threshold
                                if 'thresholds' in config and not isinstance(config['thresholds'], (list, tuple)):
                                    config['thresholds'] = [
                                        config['thresholds']]
                            elif 'threshold' in config:
                                config['threshold'] = threshold
                            compiled_metrics.append(
                                m.__class__.from_config(config))
                        else:
                            compiled_metrics.append(m)

                    model_eval.compile(
                        optimizer=optimizer, loss=loss_instance, metrics=compiled_metrics)
                    print("Provided model compiled.")

        # 3. Evaluate on the TEST dataset (Metrics Calculation)
        print("Evaluating model on TEST set (calculating metrics)...")
        evaluation_results = model_eval.evaluate(
            test_dataset,
            verbose=1,
            return_dict=True
        )

        # 4. Measure Inference Time Separately
        print("\nMeasuring inference time on TEST set...")
        total_prediction_time = 0.0
        num_batches = 0
        # Ensure we iterate from the beginning if dataset was consumed by evaluate
        test_iterator = iter(test_dataset)
        start_inference_time = time.time()
        for batch_data in test_iterator:
            images, _ = batch_data  # We only need images for prediction
            batch_start_time = time.time()
            _ = model_eval.predict_on_batch(images)  # Run prediction
            batch_end_time = time.time()
            total_prediction_time += (batch_end_time - batch_start_time)
            num_batches += 1
        end_inference_time = time.time()
        # Alternative total time measurement
        total_wall_time = end_inference_time - start_inference_time

        if num_batches > 0:
            avg_time_per_batch = total_prediction_time / num_batches
            inference_timing['total_prediction_time_s'] = total_prediction_time
            # Wall clock time for the loop
            inference_timing['total_wall_time_s'] = total_wall_time
            inference_timing['num_test_batches'] = num_batches
            inference_timing['avg_time_per_batch_s'] = avg_time_per_batch
            print(
                f"Total prediction time (sum of predict_on_batch): {total_prediction_time:.4f} seconds")
            print(
                f"Total wall clock time for prediction loop: {total_wall_time:.4f} seconds")
            print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
            if num_test_slices > 0:
                avg_time_per_slice = total_prediction_time / num_test_slices
                inference_timing['avg_time_per_slice_s'] = avg_time_per_slice
                print(
                    f"Average time per slice: {avg_time_per_slice:.6f} seconds")
            else:
                inference_timing['avg_time_per_slice_s'] = - \
                    1.0  # Indicate unknown
        else:
            print("Warning: No batches processed during inference timing.")
            inference_timing['total_prediction_time_s'] = 0.0
            inference_timing['total_wall_time_s'] = 0.0
            inference_timing['num_test_batches'] = 0
            inference_timing['avg_time_per_batch_s'] = 0.0
            inference_timing['avg_time_per_slice_s'] = 0.0

        # 5. Print Metrics and Calculate F1 Score
        print("\nKeras Evaluation Results (Test Set):")
        # Define preferred order, add 'f1_score' later
        metric_order = ['loss', 'binary_accuracy',
                        'dice_coef', 'iou', 'precision', 'recall']
        for name in metric_order:
            if name in evaluation_results:
                print(f"- {name}: {evaluation_results[name]:.4f}")
        for name, value in evaluation_results.items():
            if name not in metric_order:
                print(f"- {name}: {value:.4f}")  # Print any others

        # Calculate F1 Score
        precision_val = evaluation_results.get('precision', 0.0)
        recall_val = evaluation_results.get('recall', 0.0)
        f1_val = 0.0
        if (precision_val + recall_val) > 1e-7:
            f1_val = 2 * (precision_val * recall_val) / \
                (precision_val + recall_val)
        evaluation_results['f1_score'] = f1_val  # Add to dict
        print(f"- f1_score: {f1_val:.4f} (calculated)")
        # Add F1 to preferred print order for file
        metric_order.append('f1_score')

        # 6. Save Performance Metrics (including timing)
        try:
            perf_file_path = os.path.join(
                output_folder, "test_performances.txt")  # Save as test performance
            with open(perf_file_path, "w") as file_perf:
                file_perf.write(
                    f"Test Set Evaluation Metrics ({variant_name}):\n")
                file_perf.write("------------------------------------\n")
                for name in metric_order:  # Use updated order including F1
                    if name in evaluation_results:
                        file_perf.write(
                            f"- {name.replace('_', ' ').title()}: {evaluation_results[name]:.4f}\n")
                for name, value in evaluation_results.items():  # Add any others
                    if name not in metric_order:
                        file_perf.write(
                            f"- {name.replace('_', ' ').title()}: {value:.4f}\n")

                # Add Inference Timing
                file_perf.write("\nInference Timing (Test Set):\n")
                file_perf.write("----------------------------\n")
                if inference_timing:
                    file_perf.write(
                        f"- Total Prediction Time (sum batch): {inference_timing.get('total_prediction_time_s', 0.0):.4f} s\n")
                    file_perf.write(
                        f"- Total Wall Clock Time (loop): {inference_timing.get('total_wall_time_s', 0.0):.4f} s\n")
                    file_perf.write(
                        f"- Number of Batches: {inference_timing.get('num_test_batches', 0)}\n")
                    file_perf.write(
                        f"- Avg Time per Batch: {inference_timing.get('avg_time_per_batch_s', 0.0):.4f} s\n")
                    if num_test_slices > 0:
                        file_perf.write(
                            f"- Avg Time per Slice: {inference_timing.get('avg_time_per_slice_s', 0.0):.6f} s\n")
                    else:
                        file_perf.write(
                            "- Avg Time per Slice: N/A (slice count unknown)\n")
                    file_perf.write(
                        f"- Total Test Slices: {num_test_slices if num_test_slices > 0 else 'Unknown'}\n")
                else:
                    file_perf.write("Timing information not available.\n")

            print(f"TEST evaluation results saved to {perf_file_path}")
        except Exception as e:
            print(f"Error saving test performance metrics: {e}")

        # 7. Save Prediction Examples (from TEST dataset)
        print("\nGenerating prediction examples from TEST set...")
        # Pass the TEST dataset to the updated save function
        save_prediction_examples(
            model=model_eval,
            dataset=test_dataset,  # Use the test dataset
            output_folder=output_folder,
            num_examples=num_examples_to_save,
            threshold=threshold
        )

        print(f"--- Evaluation Finished ({variant_name}) ---")
        # Return both metrics and timing info
        return evaluation_results, inference_timing

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Return None for both if error occurs
        return None, None
    finally:
        print("Cleaning up resources after evaluation...")
        if 'test_dataset' in locals():
            del test_dataset
        # Only delete model_eval if it was created locally (not passed in)
        if 'model_eval' in locals() and model_instance is None:
            del model_eval
        gc.collect()
        print("Cleaned up evaluation resources.")


# --- Save Prediction Examples (MODIFIED for new dataset structure) ---
def save_prediction_examples(model, dataset, output_folder, num_examples=5, threshold=0.5):
    """
    Saves example predictions comparing input image and prediction overlay.
    Assumes dataset yields: (image_model_input, mask_binary_gt)
    where image_model_input is the 3-channel preprocessed image.
    """
    print(f"Saving {num_examples} comparison prediction examples...")
    # Save to test_examples subfolder
    examples_dir = os.path.join(output_folder, "test_examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Define color for binary prediction overlay
    prediction_color = 'red'

    try:
        # Take one batch from the dataset (e.g., test_dataset)
        for batch_data in dataset.take(1):
            # Unpack the batch data (expecting 2 components)
            if len(batch_data) != 2:
                print(
                    f"ERROR: Dataset expected 2 components, got {len(batch_data)}. Cannot save examples.")
                print("Expected: (image_model_input, mask_binary_gt)")
                return

            image_model_inputs, mask_binary_gts = batch_data
            batch_size = image_model_inputs.shape[0]
            print(
                f"Generating predictions for {min(num_examples, batch_size)} examples...")

            # Predict using the model input part of the batch
            predictions = model.predict(image_model_inputs)
            # Apply threshold for binary prediction map
            binary_predictions = tf.cast(
                predictions >= threshold, tf.float32).numpy()  # (B, H, W, 1)

            # Get images and GT masks for plotting
            # (B, H, W, 3) - Already processed
            image_display_inputs = image_model_inputs.numpy()
            mask_binary_gts = mask_binary_gts.numpy()       # (B, H, W, 1)

            print("Plotting and saving examples...")
            for j in range(min(num_examples, batch_size)):
                # --- Create 3 side-by-side plots ---
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Get data for the current example
                # Need to scale/clip for display if it's Z-scored
                input_img_raw = image_display_inputs[j]  # (H, W, 3)
                gt_mask_bin = mask_binary_gts[j].squeeze()  # (H, W)
                pred_binary = binary_predictions[j].squeeze()  # (H, W)

                # --- Prepare Input Image for Display ---
                # The input image is Z-scored T1c/FLAIR/T1c.
                # For visualization, we can take one channel (e.g., FLAIR - middle channel)
                # or try to rescale the 3-channel image to [0,1]. Let's use FLAIR channel.
                # Extract FLAIR channel (index 1)
                display_img = input_img_raw[:, :, 1]
                # Normalize this channel to [0, 1] for display
                min_val, max_val = np.min(display_img), np.max(display_img)
                if max_val > min_val:
                    display_img = (display_img - min_val) / (max_val - min_val)
                else:
                    display_img = np.zeros_like(
                        display_img)  # Handle constant images

                # --- Plot 1: Input Image (FLAIR channel) ---
                axes[0].imshow(display_img, cmap='gray')
                axes[0].set_title("Input Image (FLAIR channel, norm.)")
                axes[0].axis("off")

                # --- Plot 2: Ground Truth Overlay ---
                axes[1].imshow(display_img, cmap='gray', alpha=0.8)
                # Overlay GT mask (binary)
                gt_overlay = np.zeros((*gt_mask_bin.shape, 4))  # RGBA
                # Make GT green
                gt_overlay[gt_mask_bin > 0] = mcolors.to_rgba(
                    'lime', alpha=0.6)
                axes[1].imshow(gt_overlay)
                axes[1].set_title("Ground Truth (Binary)")
                axes[1].axis("off")
                gt_patch = mpatches.Patch(
                    color='lime', label='Ground Truth Tumor')
                axes[1].legend(handles=[gt_patch],
                               loc='lower right', fontsize='small')

                # --- Plot 3: Model Prediction Overlay ---
                axes[2].imshow(display_img, cmap='gray', alpha=0.8)
                # Create a colored map just for the prediction foreground
                pred_overlay = np.zeros((*pred_binary.shape, 4))  # RGBA
                pred_overlay[pred_binary > 0] = mcolors.to_rgba(
                    prediction_color, alpha=0.7)
                axes[2].imshow(pred_overlay)
                axes[2].set_title(
                    f"Model Prediction (Threshold={threshold:.2f})")
                axes[2].axis("off")
                pred_patch = mpatches.Patch(
                    color=prediction_color, label='Predicted Tumor')
                axes[2].legend(handles=[pred_patch],
                               loc='lower right', fontsize='small')

                plt.suptitle(f"Test Prediction Example {j+1}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                example_save_path = os.path.join(
                    examples_dir, f"test_comparison_example_{j+1}.png")
                plt.savefig(example_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

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
            try:
                synergy_layer = self.model.get_layer('synergy')
                alpha_val = synergy_layer.alpha.numpy()
                beta_val = synergy_layer.beta.numpy()
                print(
                    f"    Synergy weights - alpha: {alpha_val:.4f}, beta: {beta_val:.4f}")
            except ValueError:
                pass
            except Exception as e:
                print(f"    Could not get Synergy weights: {e}")
        gc.collect()

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n--- Training Finished ---")
        print(
            f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")


def lr_step_decay(epoch, lr, initial_lr, drop=0.5, epochs_drop=10):
    """Applies step decay to the learning rate. Needs initial_lr passed."""
    # This function now requires initial_lr to calculate decay correctly from start
    # The LearningRateScheduler passes the *current* lr, not the initial one.
    # The epoch number and initial_lr to calculate the scheduled lr.
    new_lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    # Ensure LR doesn't drop below a minimum value
    final_lr = max(new_lr, 1e-7)  # Example minimum LR
    # Optionally print LR changes
    # if epoch > 0 and final_lr != lr:
    #    print(f"\nEpoch {epoch+1}: LR decayed to {final_lr:.7f}")
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
            if key != 'loss' and f'val_{key}' in history_dict:
                title = key.replace('_', ' ').title()
                if 'dice_coef' in key:
                    title = 'Dice Coefficient'
                elif 'iou' in key:
                    title = 'IoU'
                elif 'accuracy' in key:
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

            plt.title(f'{title}')
            plt.xlabel('Epoch')
            if metric in history_dict and val_metric in history_dict:
                plt.legend()
            if metric != 'loss':
                min_val = min(history_dict.get(
                    metric, [0]) + history_dict.get(val_metric, [0]))
                max_val = max(history_dict.get(
                    metric, [1]) + history_dict.get(val_metric, [1]))
                if min_val >= -0.05 and max_val <= 1.1:
                    plt.ylim([-0.05, 1.05])
            plt.grid(True)
            plot_index += 1

        plt.suptitle(f'{variant_name} Training History', fontsize=14)
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

# --- Completion Notification (MODIFIED) ---


def create_completion_notification(
    variant_name,
    output_folder,        # Base output folder
    completion_file,      # Full path to completion file
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
    h5_data_dir,          # Add H5 data source dir
    # split_dir,          # Add split source dir (Optional, can derive from h5_data_dir if structure is fixed)
    # Optional timing
    start_time=None,
    inference_timing=None
):
    """Creates a text file summarizing the training run and test results."""
    print("\n--- Creating Completion Notification ---")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Point to the TEST performance file
    perf_file_path = os.path.join(output_folder, "test_performances.txt")

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
                f"AS-Net ({variant_name}) Training & Testing Completed at: {timestamp}\n\n")
            f.write("Data Configuration (BraTS 2023 Preprocessed H5):\n")
            f.write(f"- H5 Slice Directory: {h5_data_dir}\n")
            # f.write(f"- Split Definition Directory: {split_dir}\n") # Optional

            f.write("\nTraining Configuration:\n")
            f.write(f"- Model Variant: {variant_name}\n")
            f.write(f"- Image dimensions (target): {img_height}x{img_width}\n")
            f.write(f"- Input Channels (from H5): {input_channels}\n")
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
            f.write(f"- Best model weights (val): {checkpoint_best_path}\n")
            f.write(f"- Output directory: {output_folder}\n")

            f.write("\n--- Final Performance Metrics (Test Set) ---\n")
            if os.path.exists(perf_file_path):
                try:
                    with open(perf_file_path, "r") as perf_file:
                        # Read only the metrics part, skip timing part if present
                        lines = perf_file.readlines()
                        for line in lines:
                            if "Inference Timing" in line:
                                break
                            f.write(line)
                except Exception as read_err:
                    f.write(
                        f"Note: Error reading test performance file ({perf_file_path}): {read_err}\n")
            else:
                f.write(
                    f"Note: Test performance file not found ({perf_file_path}). Evaluation failed or not run.\n")

            # Add Inference Timing separately
            f.write("\n--- Inference Timing (Test Set) ---\n")
            if inference_timing:
                f.write(
                    f"- Total Prediction Time (sum batch): {inference_timing.get('total_prediction_time_s', 0.0):.4f} s\n")
                f.write(
                    f"- Total Wall Clock Time (loop): {inference_timing.get('total_wall_time_s', 0.0):.4f} s\n")
                f.write(
                    f"- Number of Batches: {inference_timing.get('num_test_batches', 0)}\n")
                f.write(
                    f"- Avg Time per Batch: {inference_timing.get('avg_time_per_batch_s', 0.0):.4f} s\n")
                avg_slice_time = inference_timing.get(
                    'avg_time_per_slice_s', -1.0)
                if avg_slice_time >= 0:
                    f.write(f"- Avg Time per Slice: {avg_slice_time:.6f} s\n")
                else:
                    f.write("- Avg Time per Slice: N/A (slice count unknown)\n")
            else:
                f.write("Timing information not available.\n")

        print(f"Completion notification saved to: {completion_file}")

    except Exception as e:
        print(f"Error creating completion notification file: {e}")


# --- Final Script Cleanup Utility ---
def final_cleanup(model=None, history=None, evaluation_results=None):
    print("\n--- Final Script Cleanup ---")
    if model is not None:
        del model
    if history is not None:
        del history
    if evaluation_results is not None:
        del evaluation_results
    gc.collect()
    backend.clear_session()
    print("Script execution completed.")
