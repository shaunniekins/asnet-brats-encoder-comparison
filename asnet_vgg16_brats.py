import os
import time
import tensorflow as tf
from keras import Model, Input
from keras.applications import VGG16
from keras.layers import (
    Conv2D,
    UpSampling2D,
    concatenate,
)

# Import common components
from latest.asnet_common import (
    setup_gpu_and_mixed_precision,
    SAM, CAM, Synergy,
    DiceCoefficient, IoU,
    prepare_brats_data_gpu,
    train_model_distributed,
    evaluate_model,
    create_completion_notification,
    final_cleanup
)

# --- VGG16 Specific Constants ---
VARIANT_NAME = "VGG16"
IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_CHANNELS = 3

# --- Data Loading ---
H5_DATA_DIR = './BraTS23_preprocessed_h5_slices'

# --- Training ---
BATCH_SIZE_PER_REPLICA = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BUFFER_SIZE = 500
THRESHOLD = 0.5
USE_MIXED_PRECISION = False

# Loss Weights
COMBINED_LOSS_WEIGHTS = {
    'bce_weight': 0.5, 'dice_weight': 0.5, 'class_weight': 100.0}

# Paths
BASE_CHECKPOINT_DIR = "./BraTS23_checkpoints"
BASE_OUTPUT_DIR = "./BraTS23_output"
BASE_COMPLETION_DIR = "./BraTS23_completion_notifications"

CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, VARIANT_NAME)
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, VARIANT_NAME)
COMPLETION_FOLDER = BASE_COMPLETION_DIR

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, f"{VARIANT_NAME}_as_net_model.weights.h5")
CHECKPOINT_BEST_PATH = os.path.join(
    CHECKPOINT_DIR, f"{VARIANT_NAME}_as_net_model_best.weights.h5")
COMPLETION_FILE = os.path.join(
    COMPLETION_FOLDER, f"{VARIANT_NAME}-asnet-finished.txt")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPLETION_FOLDER, exist_ok=True)

# --- VGG16 Specific AS_Net Model Definition ---


def AS_Net_VGG16(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)):
    """Defines the AS-Net model with a VGG16 encoder."""
    inputs = Input(input_size, dtype=tf.float32)

    # VGG16's own preprocess_input is NOT used here.
    base_model = VGG16(weights="imagenet",
                       include_top=False, input_tensor=inputs)

    # --- Fine-tuning ---
    base_model.trainable = True  # Ensure base is trainable
    # Freeze earlier layers (e.g., up to block3)
    # Fine-tuning strategy might need adjustment based on performance
    freeze_until_layer = 'block4_conv1'
    print(f"Fine-tuning: Freezing layers up to {freeze_until_layer}")
    for layer in base_model.layers:
        layer.trainable = False
        if layer.name == freeze_until_layer:
            break
    # Unfreeze layers from block4 onwards
    for layer in base_model.layers:
        if not layer.trainable:  # If it was frozen above...
            continue  # Keep it frozen
        else:  # Otherwise (layers from block4 onwards), make trainable
            layer.trainable = True
            print(f" > Unfreezing layer: {layer.name}")

    # Count trainable/non-trainable
    trainable_count = sum([tf.keras.backend.count_params(w)
                          for w in base_model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w)
                              for w in base_model.non_trainable_weights])
    print(
        f'Base Model - Trainable params: {trainable_count:,} | Non-trainable params: {non_trainable_count:,}')
    # --- End Fine-tuning ---

    # Extract feature maps (Indices based on standard VGG16 structure)
    try:
        output1 = base_model.get_layer('block1_conv2').output  # (H, W, 64)
        output2 = base_model.get_layer(
            'block2_conv2').output  # (H/2, W/2, 128)
        output3 = base_model.get_layer(
            'block3_conv3').output  # (H/4, W/4, 256)
        output4 = base_model.get_layer(
            'block4_conv3').output  # (H/8, W/8, 512)
        # (H/16, W/16, 512) - Bottleneck
        output5 = base_model.get_layer('block5_conv3').output
        print("Successfully extracted VGG16 skip connections.")
    except ValueError as e:
        print(f"Error getting VGG16 layers by name: {e}")
        print("Available layer names:", [
              layer.name for layer in base_model.layers])
        raise

    # --- Decoder (using common SAM, CAM, Synergy) ---
    print("Decoder starting...")
    # Stage 1: H/16 -> H/8
    up4 = UpSampling2D((2, 2), interpolation="bilinear",
                       name='up4')(output5)  # Upsample bottleneck
    merge4 = concatenate([output4, up4], axis=-1,
                         name='merge4')  # Skip connection H/8
    filters4 = merge4.shape[-1]  # 512+512 = 1024
    print(
        f"Decoder Stage 1 (H/8): Filters={filters4}, Input Shapes: {output4.shape}, {up4.shape}")
    SAM4 = SAM(filters=filters4, name='sam4')(merge4)  # Output ~filters4/4
    CAM4 = CAM(filters=filters4, name='cam4')(merge4)  # Output ~filters4/4

    # Stage 2: H/8 -> H/4
    up_sam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    merge31 = concatenate([output3, up_sam4], axis=-1,
                          name='merge31')  # Skip H/4 (256 + ~256)
    merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
    filters3 = merge31.shape[-1]  # ~512
    print(
        f"Decoder Stage 2 (H/4): Filters={filters3}, Input Shapes: {output3.shape}, {up_sam4.shape}")
    SAM3 = SAM(filters=filters3, name='sam3')(merge31)  # Output ~filters3/4
    CAM3 = CAM(filters=filters3, name='cam3')(merge32)  # Output ~filters3/4

    # Stage 3: H/4 -> H/2
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    merge21 = concatenate([output2, up_sam3], axis=-1,
                          name='merge21')  # Skip H/2 (128 + ~128)
    merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
    filters2 = merge21.shape[-1]  # ~256
    print(
        f"Decoder Stage 3 (H/2): Filters={filters2}, Input Shapes: {output2.shape}, {up_sam3.shape}")
    SAM2 = SAM(filters=filters2, name='sam2')(merge21)  # Output ~filters2/4
    CAM2 = CAM(filters=filters2, name='cam2')(merge22)  # Output ~filters2/4

    # Stage 4: H/2 -> H
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    merge11 = concatenate([output1, up_sam2], axis=-1,
                          name='merge11')  # Skip H (64 + ~64)
    merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
    filters1 = merge11.shape[-1]  # ~128
    print(
        f"Decoder Stage 4 (H): Filters={filters1}, Input Shapes: {output1.shape}, {up_sam2.shape}")
    SAM1 = SAM(filters=filters1, name='sam1')(merge11)  # Output ~filters1/4
    CAM1 = CAM(filters=filters1, name='cam1')(merge12)  # Output ~filters1/4

    # Synergy
    synergy_output = Synergy(name='synergy')([SAM1, CAM1])
    print(f"Synergy Output Shape: {synergy_output.shape}")

    # Final output layer (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output', dtype='float32')(synergy_output)
    print(f"Final Output Shape: {output.shape}")

    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{VARIANT_NAME}')
    return model


if __name__ == "__main__":
    script_start_time = time.time()

    # Setup GPU and mixed precision
    strategy, global_batch_size = setup_gpu_and_mixed_precision(
        batch_size_per_replica=BATCH_SIZE_PER_REPLICA,
        use_mixed_precision=USE_MIXED_PRECISION
    )

    # Define metrics list (instantiate those needing threshold)
    metrics_list = [
        'binary_accuracy',
        DiceCoefficient(threshold=THRESHOLD, name='dice_coef'),
        IoU(threshold=THRESHOLD, name='iou'),
        tf.keras.metrics.Precision(
            thresholds=THRESHOLD, name="precision"),  # Use thresholds plural
        # Use thresholds plural
        tf.keras.metrics.Recall(thresholds=THRESHOLD, name="recall")
    ]

    # --- Train ---
    model_instance = None
    history = None
    inference_timing_results = None
    if not os.path.exists(COMPLETION_FILE):
        print(
            f"\nCompletion file {COMPLETION_FILE} not found. Starting training...")
        history, model_instance = train_model_distributed(
            model_func=AS_Net_VGG16,
            dataset_func=prepare_brats_data_gpu,
            strategy=strategy,
            global_batch_size=global_batch_size,
            # Pass constants
            variant_name=VARIANT_NAME,
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
            h5_data_dir=H5_DATA_DIR,
            buffer_size=BUFFER_SIZE,
        )
    else:
        print(f"\nCompletion file {COMPLETION_FILE} found. Skipping training.")

    # --- Evaluate on TEST set ---
    print("\nStarting evaluation on TEST set...")
    evaluation_results, inference_timing_results = evaluate_model(
        model_func=AS_Net_VGG16,
        dataset_func=prepare_brats_data_gpu,
        strategy=strategy,
        global_batch_size=global_batch_size,
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
        h5_data_dir=H5_DATA_DIR,
        model_instance=model_instance,
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
        h5_data_dir=H5_DATA_DIR,
        start_time=script_start_time,
        inference_timing=inference_timing_results
    )

    # --- Cleanup ---
    final_cleanup(model=model_instance, history=history,
                  evaluation_results=evaluation_results)
