# asnet_efficientnetv2_brats.py

import os
import time
import tensorflow as tf
from keras import Model, Input
from keras.applications import (
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3
)
from keras.applications import efficientnet_v2 as efficientnet_v2_preprocessing
from keras.layers import (
    Conv2D,
    UpSampling2D,
    concatenate  # Keep for Synergy module
)

# Import common components
from asnet_common import (
    setup_gpu_and_mixed_precision,
    ResizeToMatchLayer,  # Needed for EfficientNet decoder
    SAM, CAM, Synergy,  # Core AS-Net modules
    DiceCoefficient, IoU,  # Metrics
    prepare_brats_data_gpu,
    train_model_distributed,
    evaluate_model,
    create_completion_notification,
    final_cleanup
)

# --- EfficientNetV2 Specific Constants ---
# Select the EfficientNetV2 variant by uncommenting ONE line:
EFFICIENTNET_VARIANT = 'EfficientNetV2B0'
# EFFICIENTNET_VARIANT = 'EfficientNetV2B1'
# EFFICIENTNET_VARIANT = 'EfficientNetV2B2'
# EFFICIENTNET_VARIANT = 'EfficientNetV2B3'

VARIANT_NAME = EFFICIENTNET_VARIANT  # e.g., "EfficientNetV2B0"
VARIANT_SUFFIX = EFFICIENTNET_VARIANT.lower()  # e.g., "efficientnetv2b0"

# Input Dimensions based on Variant
if EFFICIENTNET_VARIANT == 'EfficientNetV2B0':
    IMG_HEIGHT, IMG_WIDTH = 224, 224
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B1':
    IMG_HEIGHT, IMG_WIDTH = 240, 240
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B2':
    IMG_HEIGHT, IMG_WIDTH = 260, 260
elif EFFICIENTNET_VARIANT == 'EfficientNetV2B3':
    IMG_HEIGHT, IMG_WIDTH = 300, 300
else:
    raise ValueError(
        f"Unsupported EfficientNetV2 variant: {EFFICIENTNET_VARIANT}")

INPUT_CHANNELS = 3  # EfficientNetV2 needs 3 channels

# Data Loading
MODALITY_INDICES = [1, 3]  # T1ce, FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES)
# Map loaded [T1ce, FLAIR] -> [T1ce, FLAIR, T1ce] for RGB
RGB_MAPPING_INDICES = [0, 1, 0]

# Training
BATCH_SIZE_PER_REPLICA = 4  # Adjust based on variant and GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BUFFER_SIZE = 300
THRESHOLD = 0.5
USE_MIXED_PRECISION = False  # Set to True if desired (check GPU compatibility)

# Loss Weights
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5,
                         'dice_weight': 0.5, 'class_weight': 100.0}

# Paths (use VARIANT_SUFFIX)
CHECKPOINT_DIR = f"./{VARIANT_SUFFIX}-checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{VARIANT_SUFFIX}_as_net_model.weights.h5"
CHECKPOINT_BEST_PATH = f"{CHECKPOINT_DIR}/{VARIANT_SUFFIX}_as_net_model_best.weights.h5"
OUTPUT_DIR = f"{VARIANT_SUFFIX}-output"
COMPLETION_FILE = f"{VARIANT_SUFFIX}-asnet-finished-training.txt"
DATASET_PATH = "brats2020-training-data/"  # Adjust if needed
METADATA_FILE = os.path.join(DATASET_PATH, "BraTS20 Training Metadata.csv")
H5_DATA_DIR = os.path.join(
    DATASET_PATH, "BraTS2020_training_data/content/data")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "examples"), exist_ok=True)


# --- EfficientNetV2 Specific AS_Net Model Definition ---
def AS_Net_EfficientNetV2(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), variant='EfficientNetV2B0'):
    """Defines the AS-Net model with an EfficientNetV2 encoder."""
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Apply EfficientNetV2 preprocessing INSIDE the model
    preprocessed_inputs = efficientnet_v2_preprocessing.preprocess_input(
        inputs)

    # Load Base Model & Define Skips/Fine-tuning based on variant
    if variant == 'EfficientNetV2B0':
        print("Loading EfficientNetV2B0 encoder...")
        base_model = EfficientNetV2B0(
            weights="imagenet", include_top=False, input_tensor=preprocessed_inputs)
        # Skips for B0 (Verify with model.summary()) - Based on 224 input
        skip_layer_names = [
            'block1a_project_activation',  # ~112x112
            'block2b_add',                # ~56x56
            'block3b_add',                # ~28x28
            'block5e_add',                # ~14x14
            'top_activation'              # ~7x7 (Bottleneck)
        ]
        fine_tune_start_block = 4  # Start unfreezing from block 4 or 5
    elif variant == 'EfficientNetV2B1':
        print("Loading EfficientNetV2B1 encoder...")
        base_model = EfficientNetV2B1(
            weights="imagenet", include_top=False, input_tensor=preprocessed_inputs)
        # Skips for B1 (Verify with model.summary()) - Based on 240 input
        skip_layer_names = [
            'block1a_project_activation',  # ~120x120
            'block2c_add',                # ~60x60
            'block3c_add',                # ~30x30
            'block5f_add',                # ~15x15
            'top_activation'              # ~8x8 (Bottleneck)
        ]
        fine_tune_start_block = 4
    elif variant == 'EfficientNetV2B2':
        print("Loading EfficientNetV2B2 encoder...")
        base_model = EfficientNetV2B2(
            weights="imagenet", include_top=False, input_tensor=preprocessed_inputs)
        # Skips for B2 (Verify with model.summary()) - Based on 260 input
        skip_layer_names = [
            'block1a_project_activation',  # ~130x130
            'block2c_add',                # ~65x65
            'block3c_add',                # ~33x33 (~H/8)
            'block5g_add',                # ~17x17 (~H/16)
            'top_activation'              # ~9x9 (~H/32 - Bottleneck)
        ]
        fine_tune_start_block = 4
    elif variant == 'EfficientNetV2B3':
        print("Loading EfficientNetV2B3 encoder...")
        base_model = EfficientNetV2B3(
            weights="imagenet", include_top=False, input_tensor=preprocessed_inputs)
        # Skips for B3 (Verify with model.summary()) - Based on 300 input
        skip_layer_names = [
            'block1a_project_activation',  # ~150x150
            'block2d_add',                # ~75x75
            'block3d_add',                # ~38x38
            'block5j_add',                # ~19x19
            'top_activation'              # ~10x10 (Bottleneck)
        ]
        fine_tune_start_block = 4
    else:
        raise ValueError(f"Unsupported EfficientNetV2 variant: {variant}")

    # --- Fine-tuning ---
    base_model.trainable = True
    freeze_until_block_name = f'block{fine_tune_start_block}'
    print(
        f"Fine-tuning: Freezing layers up to block '{freeze_until_block_name}'...")
    layer_found = False
    for layer in base_model.layers:
        if freeze_until_block_name in layer.name:
            layer_found = True
            print(
                f"Reached block '{freeze_until_block_name}' at layer: {layer.name}. Unfreezing subsequent layers.")
        if not layer_found:
            layer.trainable = False
        else:
            # Optional: Keep BN frozen during fine-tuning
            # if isinstance(layer, BatchNormalization): layer.trainable = False
            # else: layer.trainable = True
            layer.trainable = True  # Unfreeze layers from the target block onwards

    if not layer_found:
        print(
            f"Warning: Block '{freeze_until_block_name}' not found. Fine-tuning all layers.")
        for layer in base_model.layers:
            layer.trainable = True

    trainable_count = sum([tf.keras.backend.count_params(w)
                          for w in base_model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w)
                              for w in base_model.non_trainable_weights])
    print(
        f'Base Model - Trainable params: {trainable_count:,} | Non-trainable params: {non_trainable_count:,}')
    # --- End Fine-tuning ---

    # Extract features, handling potential errors
    encoder_outputs = []
    print("Extracting skip connections from layers:")
    for name in skip_layer_names:
        try:
            layer_output = base_model.get_layer(name).output
            encoder_outputs.append(layer_output)
            print(f" - {name}: Shape {layer_output.shape}")
        except ValueError as e:
            print(
                f"\nERROR: Could not find layer '{name}' in {variant}. Error: {e}")
            print("Available layers (first 60):")
            for i, layer in enumerate(base_model.layers):
                shape_str = f" - Output Shape: {layer.output_shape}" if hasattr(
                    layer, 'output_shape') else ""
                print(f"  {i}: {layer.name}{shape_str}")
                if i > 60:
                    break
            raise ValueError(
                f"Layer {name} not found. Check skip_layer_names for {variant}.")

    if len(encoder_outputs) != 5:
        raise ValueError(
            f"Expected 5 skip connections, but got {len(encoder_outputs)} for {variant}")
    output1, output2, output3, output4, bottleneck = encoder_outputs
    # Example shapes for B2/260: (130,130), (65,65), (33,33), (17,17), (9,9)

    # --- Decoder (using common SAM, CAM, Synergy, ResizeToMatchLayer) ---
    # Handles potential odd dimensions via ResizeToMatchLayer
    print("\nDecoder starting...")

    # Stage 1: Bottleneck (~H/32) -> ~H/16
    up4 = UpSampling2D(size=(2, 2), interpolation="bilinear",
                       name='up_bottleneck')(bottleneck)
    # Use custom layer to precisely match spatial dims of skip connection output4
    up4_resized = ResizeToMatchLayer(name='resize_up4')([up4, output4])
    merge4 = concatenate([output4, up4_resized], axis=-1, name='merge4')
    filters4 = merge4.shape[-1]
    print(
        f"Decoder Stage 1 (~H/16): Filters={filters4}, Input Shapes: {output4.shape}, {up4_resized.shape}")
    SAM4 = SAM(filters=filters4, name='sam4')(merge4)
    CAM4 = CAM(filters=filters4, name='cam4')(merge4)

    # Stage 2: ~H/16 -> ~H/8
    up_sam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    up_sam4_resized = ResizeToMatchLayer(
        name='resize_up_sam4')([up_sam4, output3])
    up_cam4_resized = ResizeToMatchLayer(
        name='resize_up_cam4')([up_cam4, output3])
    merge31 = concatenate([output3, up_sam4_resized], axis=-1, name='merge31')
    merge32 = concatenate([output3, up_cam4_resized], axis=-1, name='merge32')
    filters3 = merge31.shape[-1]
    print(
        f"Decoder Stage 2 (~H/8): Filters={filters3}, Input Shapes: {output3.shape}, {up_sam4_resized.shape}")
    SAM3 = SAM(filters=filters3, name='sam3')(merge31)
    CAM3 = CAM(filters=filters3, name='cam3')(merge32)

    # Stage 3: ~H/8 -> ~H/4
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    up_sam3_resized = ResizeToMatchLayer(
        name='resize_up_sam3')([up_sam3, output2])
    up_cam3_resized = ResizeToMatchLayer(
        name='resize_up_cam3')([up_cam3, output2])
    merge21 = concatenate([output2, up_sam3_resized], axis=-1, name='merge21')
    merge22 = concatenate([output2, up_cam3_resized], axis=-1, name='merge22')
    filters2 = merge21.shape[-1]
    print(
        f"Decoder Stage 3 (~H/4): Filters={filters2}, Input Shapes: {output2.shape}, {up_sam3_resized.shape}")
    SAM2 = SAM(filters=filters2, name='sam2')(merge21)
    CAM2 = CAM(filters=filters2, name='cam2')(merge22)

    # Stage 4: ~H/4 -> ~H/2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    up_sam2_resized = ResizeToMatchLayer(
        name='resize_up_sam2')([up_sam2, output1])
    up_cam2_resized = ResizeToMatchLayer(
        name='resize_up_cam2')([up_cam2, output1])
    merge11 = concatenate([output1, up_sam2_resized], axis=-1, name='merge11')
    merge12 = concatenate([output1, up_cam2_resized], axis=-1, name='merge12')
    filters1 = merge11.shape[-1]
    print(
        f"Decoder Stage 4 (~H/2): Filters={filters1}, Input Shapes: {output1.shape}, {up_sam2_resized.shape}")
    SAM1 = SAM(filters=filters1, name='sam1')(merge11)
    CAM1 = CAM(filters=filters1, name='cam1')(merge12)

    # Stage 5: ~H/2 -> H (Final Upsample)
    # Target final size is the input size
    target_h, target_w = input_size[0], input_size[1]
    final_up_sam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
    final_up_cam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
    # Final resize to ensure exact output dimensions HxW
    final_up_sam_resized = tf.image.resize(
        final_up_sam, [target_h, target_w], method='bilinear')
    final_up_cam_resized = tf.image.resize(
        final_up_cam, [target_h, target_w], method='bilinear')
    print(
        f"Final Upsample (H): SAM/CAM Shapes after resize: {final_up_sam_resized.shape}")

    # Synergy
    synergy_output = Synergy(name='synergy')(
        [final_up_sam_resized, final_up_cam_resized])
    print(f"Synergy Output Shape: {synergy_output.shape}")

    # Final output layer (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output', dtype='float32')(synergy_output)
    print(f"Final Output Shape: {output.shape}")

    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{variant}')
    return model


# --- Main Execution Block ---
if __name__ == "__main__":
    script_start_time = time.time()

    # Setup GPU and mixed precision
    strategy, global_batch_size = setup_gpu_and_mixed_precision(
        batch_size_per_replica=BATCH_SIZE_PER_REPLICA,
        use_mixed_precision=USE_MIXED_PRECISION
    )

    # Define metrics list
    metrics_list = [
        'binary_accuracy',
        DiceCoefficient(threshold=THRESHOLD, name='dice_coef'),
        IoU(threshold=THRESHOLD, name='iou'),
        tf.keras.metrics.Precision(thresholds=THRESHOLD, name="precision"),
        tf.keras.metrics.Recall(thresholds=THRESHOLD, name="recall")
    ]

    # --- Train ---
    model_instance = None
    history = None
    if not os.path.exists(COMPLETION_FILE):
        print(
            f"\nCompletion file {COMPLETION_FILE} not found. Starting training...")
        history, model_instance = train_model_distributed(
            model_func=AS_Net_EfficientNetV2,  # Use the EfficientNetV2 specific function
            dataset_func=prepare_brats_data_gpu,
            strategy=strategy,
            global_batch_size=global_batch_size,
            # Pass constants
            variant_name=VARIANT_NAME,  # e.g., "EfficientNetV2B0"
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_best_path=CHECKPOINT_BEST_PATH,
            output_dir=OUTPUT_DIR,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            input_channels=INPUT_CHANNELS,
            num_epochs=NUM_EPOCHS,
            initial_learning_rate=LEARNING_RATE,
            combined_loss_weights=COMBINED_LOSS_WEIGHTS,
            metrics_list=metrics_list,
            threshold=THRESHOLD,
            metadata_file=METADATA_FILE,
            h5_data_dir=H5_DATA_DIR,
            buffer_size=BUFFER_SIZE,
            modality_indices=MODALITY_INDICES,
            rgb_mapping_indices=RGB_MAPPING_INDICES,
            # Pass model specific kwargs
            variant=EFFICIENTNET_VARIANT  # Pass 'EfficientNetV2B0', etc.
        )
    else:
        print(f"\nCompletion file {COMPLETION_FILE} found. Skipping training.")

    # --- Evaluate ---
    print("\nStarting evaluation...")
    evaluation_results = evaluate_model(
        model_func=AS_Net_EfficientNetV2,  # Use the EfficientNetV2 specific function
        dataset_func=prepare_brats_data_gpu,
        strategy=strategy,
        global_batch_size=global_batch_size,
        # Pass constants
        variant_name=VARIANT_NAME,
        checkpoint_best_path=CHECKPOINT_BEST_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_folder=OUTPUT_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        input_channels=INPUT_CHANNELS,
        threshold=THRESHOLD,
        loss_config=COMBINED_LOSS_WEIGHTS,
        metrics_list=metrics_list,
        metadata_file=METADATA_FILE,
        h5_data_dir=H5_DATA_DIR,
        modality_indices=MODALITY_INDICES,
        rgb_mapping_indices=RGB_MAPPING_INDICES,
        # Pass trained model instance if available
        model_instance=model_instance,
        # Pass model specific kwargs
        variant=EFFICIENTNET_VARIANT  # Pass 'EfficientNetV2B0', etc.
    )

    # --- Notify ---
    create_completion_notification(
        variant_name=VARIANT_NAME,
        output_folder=OUTPUT_DIR,
        completion_file=COMPLETION_FILE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        input_channels=INPUT_CHANNELS,
        batch_size_per_replica=BATCH_SIZE_PER_REPLICA,
        global_batch_size=global_batch_size,
        num_epochs=NUM_EPOCHS,
        initial_learning_rate=LEARNING_RATE,
        loss_config=COMBINED_LOSS_WEIGHTS,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_best_path=CHECKPOINT_BEST_PATH,
        start_time=script_start_time
    )

    # --- Cleanup ---
    final_cleanup(model=model_instance, history=history,
                  evaluation_results=evaluation_results)
