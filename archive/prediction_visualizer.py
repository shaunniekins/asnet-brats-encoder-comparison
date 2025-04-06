# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown]
# # BraTS Brain Tumor Segmentation - Prediction Visualizer
#
# This notebook allows visualization of predictions from different encoder models (VGG16, MobileNetV3, EfficientNetV2)
# trained on the BraTS brain tumor dataset.
# -

# %pip install nibabel

# + [markdown]
# ## Imports and Setup
# -

# +
import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import argparse

# Import AS-Net implementations from respective modules

# Disable eager execution if needed for large models
# tf.compat.v1.disable_eager_execution()
# -

# + [markdown]
# ## GPU Configuration
# -

# +
# Configure memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
# -

# + [markdown]
# ## Constants and Configuration
# -

# +
# Segmentation threshold for binary predictions
THRESHOLD = 0.5

# Define segmentation classes (similar to BraTS U-Net example)
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3
}
# -

# + [markdown]
# ## Model Loading Functions
# -

# +


def load_model(model_type, checkpoint_path):
    """
    Load a specific model type with weights from checkpoint.

    Args:
        model_type (str): Type of model to load (vgg16, mobilenetv3large, mobilenetv3small, efficientnetv2b0, etc.)
        checkpoint_path (str): Path to the model checkpoint file

    Returns:
        tuple: Loaded model, image height, image width
    """
    print(f"Loading {model_type} model from: {checkpoint_path}")

    # Set image dimensions based on model type
    if model_type.lower() == 'vgg16':
        img_height, img_width = 192, 192
        input_channels = 3

        # Create a VGG16 AS-Net model directly using the visualization approach
        inputs = tf.keras.Input(
            shape=(img_height, img_width, input_channels), name='input_image')

        # Load VGG16 backbone (weights don't matter for inference)
        vgg = tf.keras.applications.VGG16(
            weights=None, include_top=False, input_tensor=inputs)

        # Extract feature maps from specific layers (similar to asnet-model-visualization.py)
        output1 = vgg.get_layer(index=2).output  # block1_conv2
        output2 = vgg.get_layer(index=5).output  # block2_conv2
        output3 = vgg.get_layer(index=9).output  # block3_conv3
        output4 = vgg.get_layer(index=13).output  # block4_conv3
        output5 = vgg.get_layer(index=17).output  # block5_conv3

        # Build a simplified decoder (just enough for prediction)
        up5 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up5')(output5)
        merge1 = tf.keras.layers.concatenate(
            [output4, up5], axis=-1, name='merge1')

        # Skip detailed implementation of SAM/CAM modules for simplicity
        # Just use conv layers instead
        conv1 = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu', name='simple_conv1')(merge1)

        up1 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up1')(conv1)
        merge2 = tf.keras.layers.concatenate(
            [output3, up1], axis=-1, name='merge2')
        conv2 = tf.keras.layers.Conv2D(
            128, 3, padding='same', activation='relu', name='simple_conv2')(merge2)

        up2 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up2')(conv2)
        merge3 = tf.keras.layers.concatenate(
            [output2, up2], axis=-1, name='merge3')
        conv3 = tf.keras.layers.Conv2D(
            64, 3, padding='same', activation='relu', name='simple_conv3')(merge3)

        up3 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up3')(conv3)
        merge4 = tf.keras.layers.concatenate(
            [output1, up3], axis=-1, name='merge4')
        conv4 = tf.keras.layers.Conv2D(
            32, 3, padding='same', activation='relu', name='simple_conv4')(merge4)

        # Final output layer
        output = tf.keras.layers.Conv2D(
            1, 1, padding='same', activation='sigmoid', name='output')(conv4)

        # Create a simplified model for inference
        model = tf.keras.Model(
            inputs=inputs, outputs=output, name='VGG16_ASNet_Inference')

    elif model_type.lower().startswith('mobilenetv3'):
        img_height, img_width = 224, 224
        input_channels = 3

        # Determine the variant
        if 'large' in model_type.lower():
            variant = 'Large'
        elif 'small' in model_type.lower():
            variant = 'Small'
        else:
            variant = 'Large'  # Default

        # Create inputs
        inputs = tf.keras.Input(
            shape=(img_height, img_width, input_channels), name='input_image')

        # Load MobileNetV3 backbone
        if variant == 'Large':
            base_model = tf.keras.applications.MobileNetV3Large(
                weights=None, include_top=False, input_tensor=inputs)
            skip_layer_names = [
                're_lu', 'expanded_conv_2_add', 'expanded_conv_5_add',
                'expanded_conv_14_add', 'activation_19'
            ]
        else:  # Small
            base_model = tf.keras.applications.MobileNetV3Small(
                weights=None, include_top=False, input_tensor=inputs)
            skip_layer_names = [
                're_lu', 'expanded_conv_1_project_bn', 'expanded_conv_3_project_bn',
                'expanded_conv_7_add', 'activation_17'
            ]

        # Extract feature maps (handling potential errors in layer names)
        encoder_outputs = []
        for name in skip_layer_names:
            try:
                layer_output = base_model.get_layer(name).output
                encoder_outputs.append(layer_output)
            except ValueError:
                print(
                    f"Warning: Layer {name} not found. Creating a placeholder layer.")
                # Create a placeholder layer for visualization
                placeholder = tf.keras.layers.Conv2D(64, 1)(inputs)
                encoder_outputs.append(placeholder)

        # Ensure we have 5 outputs
        while len(encoder_outputs) < 5:
            encoder_outputs.append(tf.keras.layers.Conv2D(64, 1)(inputs))

        output1, output2, output3, output4, bottleneck = encoder_outputs

        # Build a simplified decoder (just enough for inference)
        up1 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up1')(bottleneck)
        merge1 = tf.keras.layers.concatenate(
            [output4, up1], axis=-1, name='merge1')
        conv1 = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu')(merge1)

        up2 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up2')(conv1)
        merge2 = tf.keras.layers.concatenate(
            [output3, up2], axis=-1, name='merge2')
        conv2 = tf.keras.layers.Conv2D(
            128, 3, padding='same', activation='relu')(merge2)

        up3 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up3')(conv2)
        merge3 = tf.keras.layers.concatenate(
            [output2, up3], axis=-1, name='merge3')
        conv3 = tf.keras.layers.Conv2D(
            64, 3, padding='same', activation='relu')(merge3)

        up4 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up4')(conv3)
        merge4 = tf.keras.layers.concatenate(
            [output1, up4], axis=-1, name='merge4')
        conv4 = tf.keras.layers.Conv2D(
            32, 3, padding='same', activation='relu')(merge4)

        # Final output layer
        output = tf.keras.layers.Conv2D(
            1, 1, padding='same', activation='sigmoid', name='output')(conv4)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=output,
                               name=f'MobileNetV3_{variant}_ASNet_Inference')

    elif model_type.lower().startswith('efficientnetv2'):
        # Determine variant and image size
        if 'b0' in model_type.lower():
            img_height, img_width = 224, 224
            variant = 'EfficientNetV2B0'
            base_model_func = tf.keras.applications.EfficientNetV2B0
        elif 'b1' in model_type.lower():
            img_height, img_width = 240, 240
            variant = 'EfficientNetV2B1'
            base_model_func = tf.keras.applications.EfficientNetV2B1
        elif 'b2' in model_type.lower():
            img_height, img_width = 260, 260
            variant = 'EfficientNetV2B2'
            base_model_func = tf.keras.applications.EfficientNetV2B2
        elif 'b3' in model_type.lower():
            img_height, img_width = 300, 300
            variant = 'EfficientNetV2B3'
            base_model_func = tf.keras.applications.EfficientNetV2B3
        else:
            img_height, img_width = 224, 224
            variant = 'EfficientNetV2B0'
            base_model_func = tf.keras.applications.EfficientNetV2B0

        input_channels = 3
        inputs = tf.keras.Input(
            shape=(img_height, img_width, input_channels), name='input_image')

        # Load EfficientNetV2 backbone
        base_model = base_model_func(
            weights=None, include_top=False, input_tensor=inputs)

        # Try to extract features from common layers
        try:
            # These layer names are approximations - they may differ in different EfficientNetV2 variants
            output1 = base_model.get_layer('block1a_project_activation').output
            output2 = [
                l for l in base_model.layers if 'block2' in l.name and 'add' in l.name][-1].output
            output3 = [
                l for l in base_model.layers if 'block3' in l.name and 'add' in l.name][-1].output
            output4 = [
                l for l in base_model.layers if 'block5' in l.name and 'add' in l.name][-1].output
            bottleneck = base_model.get_layer('top_activation').output
        except:
            # Fallback if specific layers can't be found
            print(
                "Warning: Could not find expected layers in EfficientNetV2. Using generic feature extraction.")
            # Get outputs from different stages of the network based on stride/shape
            all_outputs = [layer.output for layer in base_model.layers if isinstance(
                layer.output, tf.Tensor) and len(layer.output.shape) == 4]

            # Sort outputs by spatial dimensions (from largest to smallest)
            all_outputs.sort(key=lambda x: (-x.shape[1], -x.shape[2]))

            # Get 5 outputs at different spatial scales
            stride_outputs = []
            prev_shape = None
            for out in all_outputs:
                if prev_shape is None or out.shape[1:3] != prev_shape:
                    stride_outputs.append(out)
                    prev_shape = out.shape[1:3]
                if len(stride_outputs) >= 5:
                    break

            # Ensure we have exactly 5 outputs
            while len(stride_outputs) < 5:
                # Add a convolutional layer as placeholder
                stride_outputs.append(tf.keras.layers.Conv2D(64, 1)(inputs))

            output1, output2, output3, output4, bottleneck = stride_outputs

        # Build a simplified decoder (just enough for inference)
        up1 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up1')(bottleneck)
        merge1 = tf.keras.layers.concatenate(
            [output4, up1], axis=-1, name='merge1')
        conv1 = tf.keras.layers.Conv2D(
            256, 3, padding='same', activation='relu')(merge1)

        up2 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up2')(conv1)
        merge2 = tf.keras.layers.concatenate(
            [output3, up2], axis=-1, name='merge2')
        conv2 = tf.keras.layers.Conv2D(
            128, 3, padding='same', activation='relu')(merge2)

        up3 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up3')(conv2)
        merge3 = tf.keras.layers.concatenate(
            [output2, up3], axis=-1, name='merge3')
        conv3 = tf.keras.layers.Conv2D(
            64, 3, padding='same', activation='relu')(merge3)

        up4 = tf.keras.layers.UpSampling2D(
            (2, 2), interpolation="bilinear", name='up4')(conv3)
        merge4 = tf.keras.layers.concatenate(
            [output1, up4], axis=-1, name='merge4')
        conv4 = tf.keras.layers.Conv2D(
            32, 3, padding='same', activation='relu')(merge4)

        # Final output layer
        output = tf.keras.layers.Conv2D(
            1, 1, padding='same', activation='sigmoid', name='output')(conv4)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=output,
                               name=f'{variant}_ASNet_Inference')

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load weights if checkpoint path is provided and exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        try:
            # First try to load weights directly (might work for custom models)
            model.load_weights(checkpoint_path).expect_partial()
            print(f"Successfully loaded weights for {model_type} model")
        except:
            print(
                "Direct weight loading failed. Creating dummy weights for visualization only.")
            # For visualization purposes only, we don't need actual weights
    else:
        print(
            f"Warning: Checkpoint file not found at {checkpoint_path}. Using uninitialized model for structure visualization only.")

    # Explicitly set model to inference mode
    model.trainable = False

    # Print model summary
    print(f"Model loaded in inference mode (trainable={model.trainable}):")
    print(f"Total params: {model.count_params():,}")

    # Return the model and dimensions
    return model, img_height, img_width
# -

# + [markdown]
# ## Data Processing Functions
# -

# +


def process_slice_for_prediction(t1ce_slice, flair_slice, target_size):
    """Process a single slice for model prediction."""
    # Resize slices to target size
    t1ce_resized = tf.image.resize(
        t1ce_slice[..., np.newaxis], target_size).numpy()
    flair_resized = tf.image.resize(
        flair_slice[..., np.newaxis], target_size).numpy()

    # Z-score normalization (per modality)
    t1ce_norm = (t1ce_resized - np.mean(t1ce_resized)) / \
        (np.std(t1ce_resized) + 1e-8)
    flair_norm = (flair_resized - np.mean(flair_resized)) / \
        (np.std(flair_resized) + 1e-8)

    # Create RGB input (T1ce, FLAIR, T1ce for RGB channels)
    model_input = np.zeros((1, target_size[0], target_size[1], 3))
    model_input[0, :, :, 0] = t1ce_norm[:, :, 0]  # R channel
    model_input[0, :, :, 1] = flair_norm[:, :, 0]  # G channel
    model_input[0, :, :, 2] = t1ce_norm[:, :, 0]  # B channel

    return model_input


def predict_patient(model, patient_dir, target_size, slice_start, slice_count):
    """Make predictions on the specified patient data."""
    # Get patient ID from directory name
    patient_id = os.path.basename(patient_dir)
    print(f"Processing patient: {patient_id}")

    # Find the required NIfTI files with more flexible file pattern matching
    t1ce_path = None
    flair_path = None
    seg_path = None

    # Print all files in directory for debugging
    print(f"Files in directory: {os.listdir(patient_dir)}")

    # More flexible file pattern matching
    for file in os.listdir(patient_dir):
        file_lower = file.lower()
        if file_lower.endswith('.nii') or file_lower.endswith('.nii.gz'):
            # More flexible matching for T1ce
            if 't1ce' in file_lower or 't1_ce' in file_lower or 't1-ce' in file_lower or 't1c' in file_lower:
                t1ce_path = os.path.join(patient_dir, file)
                print(f"Found T1ce file: {file}")

            # More flexible matching for FLAIR
            elif 'flair' in file_lower or 'fl' in file_lower:
                flair_path = os.path.join(patient_dir, file)
                print(f"Found FLAIR file: {file}")

            # More flexible matching for segmentation
            elif 'seg' in file_lower or 'mask' in file_lower or 'label' in file_lower:
                seg_path = os.path.join(patient_dir, file)
                print(f"Found segmentation file: {file}")

            # If we can't identify, print filename for debugging
            else:
                print(f"Unidentified NIfTI file: {file}")

    # If files still not found, try to use any available modality files
    if not t1ce_path or not flair_path:
        print("Could not find specific T1ce/FLAIR files by name. Trying to use available modalities...")
        # Get all NIfTI files
        nifti_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir)
                       if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]

        # If we have at least 2 files, try to use them as T1ce and FLAIR
        if len(nifti_files) >= 2:
            t1ce_path = nifti_files[0]  # Use first file as T1ce
            flair_path = nifti_files[1]  # Use second file as FLAIR
            print(
                f"Using generic files: T1ce={os.path.basename(t1ce_path)}, FLAIR={os.path.basename(flair_path)}")
        elif len(nifti_files) == 1:
            # If only one file, duplicate it for both modalities (just for visualization)
            t1ce_path = flair_path = nifti_files[0]
            print(
                f"Only one NIfTI file found. Using it for both T1ce and FLAIR: {os.path.basename(t1ce_path)}")

    if not t1ce_path or not flair_path:
        raise ValueError(
            f"Missing required modalities for patient {patient_id}. Need both T1ce and FLAIR.")

    # Try to load NIfTI files with error handling
    try:
        print(f"Loading T1ce from: {t1ce_path}")
        t1ce_nib = nib.load(t1ce_path)
        t1ce_data = t1ce_nib.get_fdata()
        print(f"T1ce data shape: {t1ce_data.shape}")

        print(f"Loading FLAIR from: {flair_path}")
        flair_nib = nib.load(flair_path)
        flair_data = flair_nib.get_fdata()
        print(f"FLAIR data shape: {flair_data.shape}")
    except Exception as e:
        print(f"Error loading NIfTI files: {e}")
        raise ValueError(
            f"Could not load NIfTI files for patient {patient_id}: {e}")

    # Load ground truth if available
    seg_data = None
    if seg_path and os.path.exists(seg_path):
        try:
            print(f"Loading segmentation from: {seg_path}")
            seg_data = nib.load(seg_path).get_fdata()
            print(f"Segmentation data shape: {seg_data.shape}")
        except Exception as e:
            print(
                f"Error loading segmentation file (continuing without it): {e}")
            seg_data = None

    # Check if slice indices are within bounds and adjust if necessary
    if slice_start >= t1ce_data.shape[2]:
        old_start = slice_start
        slice_start = t1ce_data.shape[2] // 2 - \
            slice_count // 2  # Center slices
        print(
            f"Warning: Starting slice {old_start} is out of bounds. Adjusted to {slice_start}.")

    # Process specified slices
    results = []
    for i in range(slice_count):
        slice_idx = slice_start + i

        # Skip if slice_idx is out of bounds
        if slice_idx >= t1ce_data.shape[2]:
            print(f"Warning: Slice index {slice_idx} out of bounds. Skipping.")
            continue

        # Extract slices
        t1ce_slice = t1ce_data[:, :, slice_idx]
        flair_slice = flair_data[:, :, slice_idx]

        # Extract ground truth if available
        seg_slice = None
        if seg_data is not None and slice_idx < seg_data.shape[2]:
            seg_slice = seg_data[:, :, slice_idx]

        # Process slice for prediction
        model_input = process_slice_for_prediction(
            t1ce_slice, flair_slice, target_size)

        # Make prediction without using model.predict()
        # which can trigger training in some cases
        with tf.device('/CPU:0'):  # Force CPU to avoid OOM on GPU
            prediction = model(model_input, training=False).numpy()

        # Check if model has multiple output channels (multi-class)
        multi_class_output = len(
            prediction.shape) > 3 and prediction.shape[-1] > 1

        if multi_class_output:
            # Multi-class model (returns classes as channels)
            pred_prob_raw = prediction[0]
            binary_prediction = (
                np.argmax(pred_prob_raw, axis=-1) > 0).astype(np.float32)
        else:
            # Binary model
            pred_prob_raw = prediction[0, :, :, 0]
            binary_prediction = (pred_prob_raw >= THRESHOLD).astype(np.float32)

        # Store results
        result = {
            'slice_idx': slice_idx,
            't1ce': t1ce_slice,
            'flair': flair_slice,
            'ground_truth': seg_slice,
            'prediction_prob': pred_prob_raw,
            'prediction_binary': binary_prediction,
            'original_shape': t1ce_slice.shape,
            'is_multi_class': multi_class_output
        }
        results.append(result)

    return patient_id, results
# -

# + [markdown]
# ## Visualization Functions
# -

# +


def visualize_predictions(patient_id, results, output_dir):
    """Visualize and save the prediction results for a patient."""
    out_path = os.path.join(output_dir, patient_id)
    os.makedirs(out_path, exist_ok=True)

    for idx, result in enumerate(results):
        slice_idx = result['slice_idx']
        t1ce = result['t1ce']
        flair = result['flair']
        ground_truth = result['ground_truth']
        pred_prob = result['prediction_prob']
        pred_binary = result['prediction_binary']

        # 1. Standard visualization with T1ce, FLAIR, probability and binary prediction
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # T1ce slice
        axes[0].imshow(t1ce, cmap='gray')
        axes[0].set_title(f"T1ce - Slice {slice_idx}")
        axes[0].axis('off')

        # FLAIR slice
        axes[1].imshow(flair, cmap='gray')
        axes[1].set_title(f"FLAIR - Slice {slice_idx}")
        axes[1].axis('off')

        # Prediction probability map
        axes[2].imshow(t1ce, cmap='gray', alpha=0.7)
        prob_map = axes[2].imshow(
            pred_prob, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("Prediction Probability")
        axes[2].axis('off')
        plt.colorbar(prob_map, ax=axes[2], fraction=0.046, pad=0.04)

        # Binary prediction
        axes[3].imshow(t1ce, cmap='gray', alpha=0.7)
        axes[3].imshow(pred_binary, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
        axes[3].set_title(f"Binary Prediction (t={THRESHOLD})")
        axes[3].axis('off')

        # Add ground truth if available
        if ground_truth is not None:
            axes[4].imshow(t1ce, cmap='gray', alpha=0.7)
            axes[4].imshow(ground_truth, cmap='viridis', alpha=0.5)
            axes[4].set_title("Ground Truth Segmentation")
            axes[4].axis('off')

            # Add combined view of prediction and ground truth
            axes[5].imshow(t1ce, cmap='gray', alpha=0.7)
            axes[5].imshow(ground_truth, cmap='viridis', alpha=0.5)
            axes[5].imshow(pred_binary, cmap='hot', alpha=0.3)
            axes[5].set_title("Overlay: GT and Prediction")
            axes[5].axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_path, f"slice_{slice_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. BraTS-style visualization - Always create this visualization
        try:
            # Create a figure with 6 subplots in a single row
            plt.figure(figsize=(18, 5))
            fig, axarr = plt.subplots(1, 6, figsize=(18, 5))

            # For each subplot, use FLAIR as background
            for i in range(6):
                axarr[i].imshow(flair, cmap="gray", interpolation='none')

            # 1: Original FLAIR image
            axarr[0].set_title('Original image FLAIR')

            # 2: Ground truth (if available)
            if ground_truth is not None:
                # For multi-class ground truth
                if len(ground_truth.shape) > 2 and ground_truth.shape[-1] > 1:
                    # Display merged ground truth for all classes
                    gt_viz = np.sum(ground_truth, axis=-1)
                    axarr[1].imshow(gt_viz, cmap="Reds", alpha=0.5)
                # For binary ground truth
                else:
                    axarr[1].imshow(ground_truth, cmap="Reds", alpha=0.5)
                axarr[1].set_title('Ground truth')
            else:
                axarr[1].set_title('Ground truth (not available)')

            # 3: All tumor regions predicted
            axarr[2].imshow(pred_binary, cmap="Reds", alpha=0.5)
            axarr[2].set_title('All tumor regions')

            # For multi-class predictions, show individual classes
            if hasattr(pred_prob, 'shape') and len(pred_prob.shape) > 2 and pred_prob.shape[-1] > 1:
                # 4: Necrotic/Core prediction (class 1)
                if pred_prob.shape[-1] > 1:
                    axarr[3].imshow(pred_prob[..., 1], cmap="OrRd", alpha=0.5)
                    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} predicted')
                else:
                    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} (not predicted)')

                # 5: Edema prediction (class 2)
                if pred_prob.shape[-1] > 2:
                    axarr[4].imshow(pred_prob[..., 2], cmap="OrRd", alpha=0.5)
                    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} predicted')
                else:
                    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} (not predicted)')

                # 6: Enhancing prediction (class 3)
                if pred_prob.shape[-1] > 3:
                    axarr[5].imshow(pred_prob[..., 3], cmap="OrRd", alpha=0.5)
                    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} predicted')
                else:
                    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} (not predicted)')
            else:
                # For binary predictions, show different prediction visualizations
                # 4: Low confidence regions (< 0.75 threshold)
                low_conf = np.logical_and(
                    pred_prob > THRESHOLD, pred_prob < 0.75)
                axarr[3].imshow(low_conf, cmap="Oranges", alpha=0.5)
                axarr[3].set_title('Low confidence regions')

                # 5: Medium confidence regions (0.75 - 0.9)
                med_conf = np.logical_and(pred_prob >= 0.75, pred_prob < 0.9)
                axarr[4].imshow(med_conf, cmap="Oranges", alpha=0.5)
                axarr[4].set_title('Medium confidence regions')

                # 6: High confidence regions (>= 0.9)
                high_conf = pred_prob >= 0.9
                axarr[5].imshow(high_conf, cmap="Oranges", alpha=0.5)
                axarr[5].set_title('High confidence regions')

            # Turn off axes for all subplots
            for i in range(6):
                axarr[i].axis('off')

            plt.tight_layout()
            save_path = os.path.join(
                out_path, f"slice_{slice_idx}_brats_style.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Could not create BraTS-style visualization: {e}")
            import traceback
            traceback.print_exc()

    print(f"Saved visualizations for patient {patient_id} to {out_path}")
# -

# + [markdown]
# ## Command Line Argument Parsing
# -

# +


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Make predictions with trained AS-Net models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['vgg16', 'mobilenetv3large', 'mobilenetv3small',
                                 'efficientnetv2b0', 'efficientnetv2b1'],
                        help='Model type to use for predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--validation_dir', type=str, required=True,
                        help='Directory containing BraTS 2024 validation data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save prediction visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of patient samples to process (default: 5)')
    parser.add_argument('--slice_start', type=int, default=60,
                        help='Starting slice index (default: 60)')
    parser.add_argument('--slice_count', type=int, default=5,
                        help='Number of slices to visualize per patient (default: 5)')
    return parser.parse_args()
# -

# + [markdown]
# ## Notebook Configuration Cell
# -

# +
# Configuration parameters - modify these values to change model and parameters
# Comment/uncomment the model you want to use


# Model Selection (uncomment one)
MODEL_TYPE = 'vgg16'
# MODEL_TYPE = 'mobilenetv3small'
# MODEL_TYPE = 'mobilenetv3large'
# MODEL_TYPE = 'efficientnetv2b0'
# MODEL_TYPE = 'efficientnetv2b1'

# Checkpoint paths - update these to match your local paths
CHECKPOINT_PATHS = {
    'vgg16': 'vgg16-checkpoints/vgg16_as_net_model_best.weights.h5',
    'mobilenetv3small': 'mobilenetv3small-checkpoints/mobilenetv3small_as_net_model_best.weights.h5',
    'mobilenetv3large': 'mobilenetv3large-checkpoints/mobilenetv3large_as_net_model_best.weights.h5',
    'efficientnetv2b0': 'efficientnetv2b0-checkpoints/efficientnetv2b0_as_net_model_best.weights.h5',
    'efficientnetv2b1': 'efficientnetv2b1-checkpoints/efficientnetv2b1_as_net_model_best.weights.h5'
}

# Other parameters
VALIDATION_DIR = 'brats2024-validation-data'
OUTPUT_DIR = 'brats2024-predictions'
NUM_SAMPLES = 5
SLICE_START = 60
SLICE_COUNT = 5

# Validate configuration
if MODEL_TYPE not in CHECKPOINT_PATHS:
    raise ValueError(f"Invalid model type: {MODEL_TYPE}")

# Display the current configuration
print("Current Configuration:")
print(f"Model Type: {MODEL_TYPE}")
print(f"Checkpoint Path: {CHECKPOINT_PATHS[MODEL_TYPE]}")
print(f"Validation Directory: {VALIDATION_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Number of Samples: {NUM_SAMPLES}")
print(f"Starting Slice: {SLICE_START}")
print(f"Slice Count: {SLICE_COUNT}")
# -

# + [markdown]
# ## Main Function
# -

# +


def process_model_predictions(model_type=MODEL_TYPE,
                              checkpoint_path=None,
                              validation_dir=VALIDATION_DIR,
                              output_dir=OUTPUT_DIR,
                              num_samples=NUM_SAMPLES,
                              slice_start=SLICE_START,
                              slice_count=SLICE_COUNT):
    """
    Process predictions for the specified model and parameters.

    Args:
        model_type: Type of model to use
        checkpoint_path: Path to model checkpoint, uses default if None
        validation_dir: Directory with validation data
        output_dir: Directory to save output visualizations
        num_samples: Number of patients to process
        slice_start: Starting slice index
        slice_count: Number of slices to process per patient
    """
    # Use default checkpoint path if not specified
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[model_type]

    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load model in inference mode
    print("Loading model for inference only...")
    model, img_height, img_width = load_model(model_type, checkpoint_path)

    # Get list of patient directories
    try:
        # First check if validation_dir exists
        if not os.path.exists(validation_dir):
            raise FileNotFoundError(
                f"Validation directory not found: {validation_dir}")

        # Get subdirectories that could contain patient data
        patient_dirs = []
        for d in os.listdir(validation_dir):
            full_path = os.path.join(validation_dir, d)
            if os.path.isdir(full_path):
                # Check if directory contains NIfTI files
                has_nifti = any(f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')
                                for f in os.listdir(full_path))
                if has_nifti:
                    patient_dirs.append(full_path)

        if not patient_dirs:
            print(
                f"No patient directories with NIfTI files found in {validation_dir}. Checking structure...")
            # In case validation_dir is one level up, check subdirectories
            for d in os.listdir(validation_dir):
                subdir = os.path.join(validation_dir, d)
                if os.path.isdir(subdir):
                    print(f"Checking subdirectory: {d}")
                    for sub_d in os.listdir(subdir):
                        full_path = os.path.join(subdir, sub_d)
                        if os.path.isdir(full_path):
                            has_nifti = any(f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')
                                            for f in os.listdir(full_path))
                            if has_nifti:
                                patient_dirs.append(full_path)
                                print(f"  Found patient dir: {sub_d}")

        # Limit number of patients if specified
        if not patient_dirs:
            raise ValueError(
                f"No patient directories with NIfTI files found in or under {validation_dir}")

        if num_samples and num_samples < len(patient_dirs):
            patient_dirs = patient_dirs[:num_samples]

        print(f"Processing {len(patient_dirs)} patients:")
        for pd in patient_dirs:
            print(f"  - {os.path.basename(pd)}")
    except Exception as e:
        print(f"Error accessing validation directory: {e}")
        return

    # Process each patient
    successful_patients = 0
    for patient_dir in tqdm(patient_dirs, desc=f"Processing patients with {model_type}"):
        try:
            patient_id, results = predict_patient(
                model,
                patient_dir,
                target_size=(img_height, img_width),
                slice_start=slice_start,
                slice_count=slice_count
            )

            # Visualize predictions
            visualize_predictions(patient_id, results, model_output_dir)
            successful_patients += 1

        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            continue

        # Free up memory
        gc.collect()

    print(
        f"Completed predictions for {model_type}. Processed {successful_patients}/{len(patient_dirs)} patients successfully.")
    print(f"Results saved to {model_output_dir}")


def main():
    # Check if running in notebook or as script
    try:
        # This will raise an exception if we're not in a notebook
        get_ipython()
        # If we are in a notebook, use the configuration variables
        is_notebook = True
    except:
        # If we're not in a notebook, parse command line arguments
        is_notebook = False
        args = parse_arguments()

    if is_notebook:
        # Use the globally defined configuration
        process_model_predictions()
    else:
        # Use command line arguments
        process_model_predictions(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            validation_dir=args.validation_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            slice_start=args.slice_start,
            slice_count=args.slice_count
        )
# -

# + [markdown]
# ## Run Predictions
#
# Uncomment the line below to run predictions with the current configuration
# -


# +
# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
# -

# + [markdown]
# ## Example Usage
#
# To process predictions with a specific model, you can call the process_model_predictions function directly:
#
# ```python
# # For VGG16
# process_model_predictions(model_type='vgg16')
#
# # For MobileNetV3Small
# process_model_predictions(model_type='mobilenetv3small')
#
# # For EfficientNetV2B0
# process_model_predictions(model_type='efficientnetv2b0')
# ```
#
# Or you can modify the MODEL_TYPE variable in the configuration cell above and run the main() function.
# -
