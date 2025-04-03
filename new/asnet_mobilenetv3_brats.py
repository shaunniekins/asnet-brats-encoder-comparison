# asnet_mobilenetv3_brats.py

import os
import time
import tensorflow as tf
from keras import Model, Input
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input
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

# --- MobileNetV3 Specific Constants ---
MOBILENET_VARIANT = 'Large'  # Choose 'Large' or 'Small'
# MOBILENET_VARIANT = 'Small'
VARIANT_NAME = f"MobileNetV3{MOBILENET_VARIANT}"
VARIANT_SUFFIX = f"mobilenetv3{MOBILENET_VARIANT.lower()}"  # For paths

IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_CHANNELS = 3  # MobileNetV3 needs 3 channels

# Data Loading
MODALITY_INDICES = [1, 3]  # T1ce, FLAIR
NUM_MODALITIES_LOADED = len(MODALITY_INDICES)
# Map loaded [T1ce, FLAIR] -> [T1ce, FLAIR, T1ce] for RGB
RGB_MAPPING_INDICES = [0, 1, 0]

# Training
BATCH_SIZE_PER_REPLICA = 4  # Adjust based on memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BUFFER_SIZE = 300
THRESHOLD = 0.5
USE_MIXED_PRECISION = False  # Set to True if you want to use mixed precision

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


# --- MobileNetV3 Specific AS_Net Model Definition ---
def AS_Net_MobileNetV3(input_size=(IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), variant='Large'):
    """Defines the AS-Net model with a MobileNetV3 encoder."""
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Apply MobileNetV3 preprocessing INSIDE the model
    preprocessed_inputs = mobilenet_preprocess_input(inputs)

    # Load Base Model
    if variant == 'Large':
        print("Loading MobileNetV3-Large encoder...")
        base_model = MobileNetV3Large(
            weights="imagenet", include_top=False, input_shape=input_size,
            input_tensor=preprocessed_inputs  # Pass preprocessed tensor
        )
        # Skip connections for Large (Verify layer names with model.summary())
        skip_layer_names = [
            # ~H/2 (112x112) Corrected from VGG script
            're_lu',
            'expanded_conv_2/add',    # ~H/4 (56x56) Corrected name
            'expanded_conv_5/add',    # ~H/8 (28x28) Corrected name
            # ~H/16 (14x14) Usually block 11 or 12 add
            'expanded_conv_11/add',
            # ~H/32 (7x7)  Final activation before pooling
            'Conv_1/act'
        ]
        fine_tune_start_layer_name = 'expanded_conv_6'  # Example: Start fine-tuning here
    elif variant == 'Small':
        print("Loading MobileNetV3-Small encoder...")
        base_model = MobileNetV3Small(
            weights="imagenet", include_top=False, input_shape=input_size,
            input_tensor=preprocessed_inputs  # Pass preprocessed tensor
        )
        # Skip connections for Small (Verify layer names with model.summary())
        skip_layer_names = [
            're_lu',                  # ~H/2 (112x112) Corrected
            # ~H/4 (56x56) Corrected (often BN after projection)
            'expanded_conv_1/project_bn',
            'expanded_conv_3/project_bn',  # ~H/8 (28x28) Corrected
            # ~H/16 (14x14) Check if block 8 or 9 add is correct
            'expanded_conv_8/add',
            'Conv_1/act'              # ~H/32 (7x7) Final activation
        ]
        fine_tune_start_layer_name = 'expanded_conv_4'  # Example: Start fine-tuning here
    else:
        raise ValueError("Invalid variant. Choose 'Large' or 'Small'.")

    # --- Fine-tuning ---
    base_model.trainable = True  # Ensure base is trainable before selective freeze
    print(
        f"Fine-tuning: Attempting to freeze layers up to layer containing '{fine_tune_start_layer_name}'...")
    layer_found = False
    for layer in base_model.layers:
        if fine_tune_start_layer_name in layer.name:
            layer_found = True
            print(
                f"Reached target layer: {layer.name}. Unfreezing subsequent layers.")
        if not layer_found:
            layer.trainable = False
        else:
            layer.trainable = True  # Unfreeze from target onwards

    if not layer_found:
        print(
            f"Warning: Start layer '{fine_tune_start_layer_name}' not found. Fine-tuning all layers.")
        for layer in base_model.layers:
            layer.trainable = True  # Fallback: train all

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
                f"\nERROR: Could not find layer '{name}' in MobileNetV3{variant}. Error: {e}")
            print("Available layers (first 50):")
            for i, layer in enumerate(base_model.layers):
                shape_str = f" - Output Shape: {layer.output_shape}" if hasattr(
                    layer, 'output_shape') else ""
                print(f"  {i}: {layer.name}{shape_str}")
                if i > 50:
                    break
            # Try finding layers with similar names
            print(f"\nLayers containing '{name.split('/')[0]}':")
            print([l.name for l in base_model.layers if name.split('/')[0] in l.name])
            raise ValueError(
                f"Layer {name} not found. Check skip_layer_names for MobileNetV3{variant}.")

    # Unpack encoder outputs (adjust based on actual number and shapes)
    if len(encoder_outputs) != 5:
        raise ValueError(
            f"Expected 5 skip connections, but got {len(encoder_outputs)} for MobileNetV3{variant}")
    output1, output2, output3, output4, bottleneck = encoder_outputs
    # Example shapes for 224x224 input: (112,112), (56,56), (28,28), (14,14), (7,7)

    # --- Decoder (using common SAM, CAM, Synergy) ---
    # Decoder structure assumes 5 stages H/2, H/4, H/8, H/16, H/32 from encoder
    # Adjust if MobileNet variants yield different structures
    print("\nDecoder starting...")

    # Stage 1: H/32 -> H/16
    up4 = UpSampling2D((2, 2), interpolation="bilinear",
                       name='up4')(bottleneck)
    merge4 = concatenate([output4, up4], axis=-1, name='merge4')
    filters4 = merge4.shape[-1]
    print(
        f"Decoder Stage 1 (H/16): Filters={filters4}, Input Shapes: {output4.shape}, {up4.shape}")
    SAM4 = SAM(filters=filters4, name='sam4')(merge4)
    CAM4 = CAM(filters=filters4, name='cam4')(merge4)

    # Stage 2: H/16 -> H/8
    up_sam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
    merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
    filters3 = merge31.shape[-1]
    print(
        f"Decoder Stage 2 (H/8): Filters={filters3}, Input Shapes: {output3.shape}, {up_sam4.shape}")
    SAM3 = SAM(filters=filters3, name='sam3')(merge31)
    CAM3 = CAM(filters=filters3, name='cam3')(merge32)

    # Stage 3: H/8 -> H/4
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
    merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
    filters2 = merge21.shape[-1]
    print(
        f"Decoder Stage 3 (H/4): Filters={filters2}, Input Shapes: {output2.shape}, {up_sam3.shape}")
    SAM2 = SAM(filters=filters2, name='sam2')(merge21)
    CAM2 = CAM(filters=filters2, name='cam2')(merge22)

    # Stage 4: H/4 -> H/2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
    merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
    filters1 = merge11.shape[-1]
    print(
        f"Decoder Stage 4 (H/2): Filters={filters1}, Input Shapes: {output1.shape}, {up_sam2.shape}")
    SAM1 = SAM(filters=filters1, name='sam1')(merge11)
    CAM1 = CAM(filters=filters1, name='cam1')(merge12)

    # Stage 5: H/2 -> H (Final Upsample)
    final_up_sam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
    final_up_cam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)
    print(f"Final Upsample (H): SAM/CAM Shapes: {final_up_sam.shape}")

    # Synergy
    synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam])
    print(f"Synergy Output Shape: {synergy_output.shape}")

    # Final output layer (use float32 for stability)
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output', dtype='float32')(synergy_output)
    print(f"Final Output Shape: {output.shape}")

    model = Model(inputs=inputs, outputs=output,
                  name=f'AS_Net_MobileNetV3_{variant}')
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
            model_func=AS_Net_MobileNetV3,  # Use the MobileNetV3 specific function
            dataset_func=prepare_brats_data_gpu,
            strategy=strategy,
            global_batch_size=global_batch_size,
            # Pass constants
            variant_name=VARIANT_NAME,  # e.g., "MobileNetV3Large"
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
            variant=MOBILENET_VARIANT  # Pass 'Large' or 'Small' to the model func
        )
    else:
        print(f"\nCompletion file {COMPLETION_FILE} found. Skipping training.")

    # --- Evaluate ---
    print("\nStarting evaluation...")
    evaluation_results = evaluate_model(
        model_func=AS_Net_MobileNetV3,  # Use the MobileNetV3 specific function
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
        variant=MOBILENET_VARIANT  # Pass 'Large' or 'Small' to the model func
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
