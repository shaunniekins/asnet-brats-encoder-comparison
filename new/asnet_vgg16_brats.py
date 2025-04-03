# asnet_vgg16_brats.py

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
from asnet_common import (
    setup_gpu_and_mixed_precision,
    SAM, CAM, Synergy,  # Core AS-Net modules
    DiceCoefficient, IoU,  # Metrics
    prepare_brats_data_gpu,
    train_model_distributed,
    evaluate_model,
    create_completion_notification,
    final_cleanup
)

# --- VGG16 Specific Constants ---
VARIANT_NAME = "VGG16"
IMG_HEIGHT = 192
IMG_WIDTH = 192
INPUT_CHANNELS = 3  # VGG16 needs 3 channels

# Data Loading
MODALITY_INDICES = [1, 3]  # T1ce, FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES)
# Map loaded [T1ce, FLAIR] -> [T1ce, FLAIR, T1ce] for RGB
RGB_MAPPING_INDICES = [0, 1, 0]

# Training
BATCH_SIZE_PER_REPLICA = 4  # Keep low for VGG16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BUFFER_SIZE = 300
THRESHOLD = 0.5
USE_MIXED_PRECISION = False  # Set to True if you want to use mixed precision

# Loss Weights (Example, tune as needed)
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5,
                         'dice_weight': 0.5, 'class_weight': 100.0}

# Paths
CHECKPOINT_DIR = "./vgg16-checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/vgg16_as_net_model.weights.h5"
CHECKPOINT_BEST_PATH = f"{CHECKPOINT_DIR}/vgg16_as_net_model_best.weights.h5"
OUTPUT_DIR = "vgg16-output"
COMPLETION_FILE = "vgg16-asnet-finished-training.txt"
DATASET_PATH = "brats2020-training-data/"  # Adjust if needed
METADATA_FILE = os.path.join(DATASET_PATH, "BraTS20 Training Metadata.csv")
H5_DATA_DIR = os.path.join(
    DATASET_PATH, "BraTS2020_training_data/content/data")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "examples"), exist_ok=True)

# --- VGG16 Specific AS_Net Model Definition ---


def AS_Net_VGG16(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS)):
    """Defines the AS-Net model with a VGG16 encoder."""
    inputs = Input(input_size, dtype=tf.float32)  # Input tensor

    # VGG16 expects inputs scaled [0, 255] and uses its own preprocess_input
    # which centers around ImageNet means. Since our data pipeline outputs Z-scored
    # data, we don't apply VGG16's preprocess_input here.
    # The Z-scoring serves as normalization.
    # If performance is poor, consider scaling Z-scored data to [0,1] or [-1,1]
    # before feeding to VGG16, or retrain VGG16's first layers.

    base_model = VGG16(weights="imagenet",
                       include_top=False, input_tensor=inputs)

    # --- Fine-tuning ---
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[11:]:
        layer.trainable = True  # Unfreeze from block4 onwards
    print("Unfroze VGG16 layers from block4 onwards for fine-tuning.")
    # --- End Fine-tuning ---

    # Extract feature maps (Verify indices with model.summary())
    output1 = base_model.get_layer(index=2).output  # block1_conv2 (H, W, 64)
    # block2_conv2 (H/2, W/2, 128)
    output2 = base_model.get_layer(index=5).output
    # block3_conv3 (H/4, W/4, 256)
    output3 = base_model.get_layer(index=9).output
    # block4_conv3 (H/8, W/8, 512)
    output4 = base_model.get_layer(index=13).output
    # block5_conv3 (H/16, W/16, 512)
    output5 = base_model.get_layer(index=17).output

    # --- Decoder (using common SAM, CAM, Synergy) ---
    # Stage 1: H/16 -> H/8
    up5 = UpSampling2D((2, 2), interpolation="bilinear", name='up5')(output5)
    # Filters = 512+512=1024
    merge1 = concatenate([output4, up5], axis=-1, name='merge1')
    SAM1 = SAM(filters=1024, name='sam1')(
        merge1)  # Output filters = 1024 // 4 = 256
    CAM1 = CAM(filters=1024, name='cam1')(merge1)  # Output filters = 256

    # Stage 2: H/8 -> H/4
    up_sam1 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam1')(SAM1)
    up_cam1 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam1')(CAM1)
    merge21 = concatenate([output3, up_sam1], axis=-1,
                          name='merge21')  # Filters = 256+256=512
    merge22 = concatenate([output3, up_cam1], axis=-1, name='merge22')
    SAM2 = SAM(filters=512, name='sam2')(
        merge21)  # Output filters = 512 // 4 = 128
    CAM2 = CAM(filters=512, name='cam2')(merge22)

    # Stage 3: H/4 -> H/2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    merge31 = concatenate([output2, up_sam2], axis=-1,
                          name='merge31')  # Filters = 128+128=256
    merge32 = concatenate([output2, up_cam2], axis=-1, name='merge32')
    SAM3 = SAM(filters=256, name='sam3')(
        merge31)  # Output filters = 256 // 4 = 64
    CAM3 = CAM(filters=256, name='cam3')(merge32)

    # Stage 4: H/2 -> H
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    merge41 = concatenate([output1, up_sam3], axis=-1,
                          name='merge41')  # Filters = 64+64=128
    merge42 = concatenate([output1, up_cam3], axis=-1, name='merge42')
    SAM4 = SAM(filters=128, name='sam4')(
        merge41)  # Output filters = 128 // 4 = 32
    CAM4 = CAM(filters=128, name='cam4')(merge42)

    # Synergy
    synergy_output = Synergy(name='synergy')(
        [SAM4, CAM4])  # Output shape (B, H, W, 1)

    # Final output layer (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output', dtype='float32')(synergy_output)

    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{VARIANT_NAME}')
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
    # Instantiate metrics that require threshold here
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
            metrics_list=metrics_list,  # Pass the list of metric objects/names
            threshold=THRESHOLD,
            metadata_file=METADATA_FILE,
            h5_data_dir=H5_DATA_DIR,
            buffer_size=BUFFER_SIZE,
            modality_indices=MODALITY_INDICES,
            rgb_mapping_indices=RGB_MAPPING_INDICES
            # No model_kwargs needed for VGG16 AS_Net definition
        )
    else:
        print(f"\nCompletion file {COMPLETION_FILE} found. Skipping training.")

    # --- Evaluate ---
    print("\nStarting evaluation...")
    evaluation_results = evaluate_model(
        model_func=AS_Net_VGG16,
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
        # Pass trained model instance if training just finished
        model_instance=model_instance
        # No model_kwargs needed for VGG16 AS_Net definition
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
