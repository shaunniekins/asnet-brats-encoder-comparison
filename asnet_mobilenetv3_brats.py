import os
import sys
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

# --- MobileNetV3 Specific Constants ---
if len(sys.argv) > 1:
    if sys.argv[1].lower() == 'large':
        MOBILENET_VARIANT = 'Large'
    elif sys.argv[1].lower() == 'small':
        MOBILENET_VARIANT = 'Small'
    else:
        print(
            f"Error: Unknown variant '{sys.argv[1]}'. Using 'Large' as default.")
        print("Usage: %run asnet_mobilenetv3_brats.py [Large|Small]")
        MOBILENET_VARIANT = 'Large'
else:
    MOBILENET_VARIANT = 'Large'

print(f"Selected MobileNetV3 variant: {MOBILENET_VARIANT}")


VARIANT_NAME = f"MobileNetV3{MOBILENET_VARIANT}_BraTS23"
VARIANT_SUFFIX = f"mobilenetv3{MOBILENET_VARIANT.lower()}"

IMG_HEIGHT = 224
IMG_WIDTH = 224
INPUT_CHANNELS = 3  # MobileNetV3 needs 3 channels

# --- Data Loading ---
H5_DATA_DIR = './preprocessed_brats23_h5_slices'

# Training
BATCH_SIZE_PER_REPLICA = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
BUFFER_SIZE = 500
THRESHOLD = 0.5
USE_MIXED_PRECISION = False

# Loss Weights
COMBINED_LOSS_WEIGHTS = {'bce_weight': 0.5,
                         'dice_weight': 0.5, 'class_weight': 100.0}

# Paths (use VARIANT_NAME)
CHECKPOINT_DIR = f"./{VARIANT_NAME}-checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{VARIANT_NAME}_as_net_model.weights.h5"
CHECKPOINT_BEST_PATH = f"{CHECKPOINT_DIR}/{VARIANT_NAME}_as_net_model_best.weights.h5"
OUTPUT_DIR = f"{VARIANT_NAME}-output"
COMPLETION_FOLDER = "completion-notifications"
COMPLETION_FILE = f"{COMPLETION_FOLDER}/{VARIANT_NAME}-asnet-finished.txt"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPLETION_FOLDER, exist_ok=True)


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
        # Updated skip connections for Large based on actual layer names
        skip_layer_names = [
            're_lu',                     # ~H/2 (112x112)
            'expanded_conv_2_add',       # ~H/4 (56x56)
            'expanded_conv_5_add',       # ~H/8 (28x28)
            'expanded_conv_11_add',      # ~H/16 (14x14)
            'activation_19'              # ~H/32 (7x7) Final activation
        ]
        fine_tune_start_layer_name = 'expanded_conv_6'
    elif variant == 'Small':
        print("Loading MobileNetV3-Small encoder...")
        base_model = MobileNetV3Small(
            weights="imagenet", include_top=False, input_shape=input_size,
            input_tensor=preprocessed_inputs
        )

        print("MobileNetV3Small Layer Names (Post Hard Restart):")
        layer_names = [layer.name for layer in base_model.layers]
        print(layer_names[:75])
        if len(layer_names) > 75:
            print(f"... and {len(layer_names)-75} more layers.")
        print(f"Actual Last layer name: {layer_names[-1]}")

        # --- CORRECTED Skip Connections for Small (Attempt 9 - SPATIAL MATCH) ---
        # Selecting layers based on expected spatial output size for U-Net decoder
        skip_layer_names = [
            # H/2 (112x112) - Layer 4 (Check shape on run)
            'activation',
            # H/4 (56x56) - Layer 16 (Check shape on run)
            'expanded_conv_project_bn',
            # H/8 (28x28) - Layer 25 (Check shape on run)
            'expanded_conv_1_project_bn',
            # H/16 (14x14) - Layer 49 (Check shape on run)
            'expanded_conv_3_project_bn',
            # H/32 (7x7) - Layer 163 (Verified OK)
            'activation_17'
        ]
        print(
            f"Using Spatially Corrected Skip Layer Names: {skip_layer_names}")
        # --- End Correction ---

        fine_tune_start_layer_name = 'expanded_conv_4'

    else:
        raise ValueError("Invalid variant. Choose 'Large' or 'Small'.")

    # --- Fine-tuning ---
    base_model.trainable = True
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
    if not os.path.exists(COMPLETION_FILE):
        print(
            f"\nCompletion file {COMPLETION_FILE} not found. Starting training...")
        history, model_instance = train_model_distributed(
            model_func=AS_Net_MobileNetV3,
            dataset_func=prepare_brats_data_gpu,
            strategy=strategy,
            global_batch_size=global_batch_size,
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
            variant=MOBILENET_VARIANT
        )
    else:
        print(f"\nCompletion file {COMPLETION_FILE} found. Skipping training.")

    # --- Evaluate on TEST set ---
    print("\nStarting evaluation on TEST set...")
    evaluation_results = evaluate_model(
        model_func=AS_Net_MobileNetV3,
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
        variant=MOBILENET_VARIANT
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
        start_time=script_start_time
    )

    # --- Cleanup ---
    final_cleanup(model=model_instance, history=history,
                  evaluation_results=evaluation_results)
