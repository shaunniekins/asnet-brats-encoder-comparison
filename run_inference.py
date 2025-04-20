import os
import time
import argparse
import tensorflow as tf
import numpy as np
import glob
import h5py
import re
from collections import defaultdict

# Import necessary components from common and model definition
from ver1_asnet_common import (
    setup_gpu_and_mixed_precision,
    save_prediction_examples,
    final_cleanup
)
# --- IMPORTANT: Import the correct model definition based on the variant ---
# Import all model variants
try:
    from ver1_asnet_vgg16_brats import AS_Net_VGG16
    from ver1_asnet_mobilenetv3_brats import AS_Net_MobileNetV3
    from ver1_asnet_efficientnetv2_brats import AS_Net_EfficientNetV2
except ImportError as e:
    print(f"Error importing model definition: {e}")
    print("Please ensure the corresponding model script exists.")
    exit(1)

# --- Model Variant Mapping (Updated to support all variants) ---
MODEL_BUILDERS = {
    'VGG16': AS_Net_VGG16,
    'MobileNetV3Small': lambda input_size: AS_Net_MobileNetV3(input_size=input_size, variant='Small'),
    'MobileNetV3Large': lambda input_size: AS_Net_MobileNetV3(input_size=input_size, variant='Large'),
    'EfficientNetV2B0': lambda input_size: AS_Net_EfficientNetV2(input_size=input_size, variant='EfficientNetV2B0'),
    'EfficientNetV2B1': lambda input_size: AS_Net_EfficientNetV2(input_size=input_size, variant='EfficientNetV2B1'),
    'EfficientNetV2B2': lambda input_size: AS_Net_EfficientNetV2(input_size=input_size, variant='EfficientNetV2B2'),
    'EfficientNetV2B3': lambda input_size: AS_Net_EfficientNetV2(input_size=input_size, variant='EfficientNetV2B3')
}

# --- Base Output Directory ---
BASE_INFERENCE_OUTPUT_DIR = "./BraTS23_inference_output"


def extract_patient_id(filename):
    """Extract patient ID from BraTS H5 slice filename."""
    match = re.match(r'(BraTS-GLI-\d+-\d+)_slice_\d+\.h5',
                     os.path.basename(filename))
    if match:
        return match.group(1)
    return "unknown"  # Fallback for unrecognized filenames


def run_inference(args):
    """Runs inference using a trained model on specified H5 data."""
    script_start_time = time.time()

    # --- Construct the final output directory ---
    # Output will be saved in BASE_INFERENCE_OUTPUT_DIR / VARIANT_NAME / USER_SPECIFIED_SUBDIR
    output_dir = os.path.join(
        BASE_INFERENCE_OUTPUT_DIR, args.variant_name, args.output_subdir)
    print(f"--- Starting Inference ({args.variant_name}) ---")
    print(f"Model Weights: {args.weights_path}")
    print(f"Inference Data Dir: {args.h5_data_dir}")
    print(f"Output Dir: {output_dir}")  # Use the constructed path
    print(f"Target Image Size: {args.img_height}x{args.img_width}")
    print(f"Batch Size (per replica): {args.batch_size_per_replica}")

    os.makedirs(output_dir, exist_ok=True)  # Use the constructed path

    # 1. Setup GPU and get strategy, global batch size
    strategy, global_batch_size = setup_gpu_and_mixed_precision(
        batch_size_per_replica=args.batch_size_per_replica,
        use_mixed_precision=args.use_mixed_precision
    )

    # 2. Prepare Inference Dataset
    print("\nPreparing inference dataset...")
    try:
        if not os.path.isdir(args.h5_data_dir):
            raise FileNotFoundError(
                f"H5 data directory not found at {args.h5_data_dir}")

        h5_files = glob.glob(os.path.join(args.h5_data_dir, "*.h5"))
        if not h5_files:
            raise ValueError(f"No H5 files found in {args.h5_data_dir}.")
        print(f"Found {len(h5_files)} H5 slice files for inference.")
        num_inference_slices = len(h5_files)

        # Group files by patient ID for later analysis
        patient_slices = defaultdict(list)
        for file_path in h5_files:
            patient_id = extract_patient_id(file_path)
            patient_slices[patient_id].append(file_path)

        num_patients = len(patient_slices)
        print(f"Found slices from {num_patients} unique patients.")

        # Create patient statistics
        patients_stats = {
            'patient_ids': list(patient_slices.keys()),
            'slice_counts': [len(slices) for slices in patient_slices.values()],
            'total_patients': num_patients
        }

        # Create a dataset directly from the file list
        inference_dataset = tf.data.Dataset.from_tensor_slices(h5_files)

        def parse_inference_h5(file_path):
            def _parse(path_tensor):
                path = path_tensor.numpy().decode("utf-8")
                try:
                    with h5py.File(path, "r") as hf:
                        image_preprocessed = hf["image"][()].astype(np.float32)
                        if "mask" in hf:
                            mask_binary = hf["mask"][()].astype(np.float32)
                        else:
                            h, w = image_preprocessed.shape[:2]
                            mask_binary = np.zeros((h, w, 1), dtype=np.float32)
                        return image_preprocessed, mask_binary
                except Exception as e:
                    print(f"Error processing file {path}: {e}")
                    dummy_h, dummy_w = 240, 240
                    dummy_image = np.zeros(
                        (dummy_h, dummy_w, args.input_channels), dtype=np.float32)
                    dummy_mask = np.zeros(
                        (dummy_h, dummy_w, 1), dtype=np.float32)
                    return dummy_image, dummy_mask

            image_data, mask_data = tf.py_function(
                _parse, [file_path], [tf.float32, tf.float32]
            )
            assumed_h, assumed_w = 240, 240
            image_data.set_shape([assumed_h, assumed_w, args.input_channels])
            mask_data.set_shape([assumed_h, assumed_w, 1])
            return image_data, mask_data

        def resize_data(image_data, mask_data):
            image_resized = tf.image.resize(
                image_data, (args.img_height, args.img_width), method='bilinear')
            image_final = tf.cast(image_resized, tf.float32)
            mask_resized = tf.image.resize(
                mask_data, (args.img_height, args.img_width), method='nearest')
            mask_final = tf.cast(mask_resized > 0.5, tf.float32)

            image_final.set_shape(
                [args.img_height, args.img_width, args.input_channels])
            mask_final.set_shape([args.img_height, args.img_width, 1])
            return image_final, mask_final

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        inference_dataset = inference_dataset.with_options(options)

        inference_dataset = (
            inference_dataset
            .map(parse_inference_h5, num_parallel_calls=tf.data.AUTOTUNE)
            .map(resize_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(global_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        print("Inference dataset prepared.")
        print("Dataset Element Spec:", inference_dataset.element_spec)

    except Exception as e:
        print(f"Error preparing inference dataset: {e}")
        return

    # 3. Build Model and Load Weights
    print("\nBuilding model architecture...")
    try:
        model_builder = MODEL_BUILDERS.get(args.variant_name)
        if not model_builder:
            raise ValueError(
                f"Unsupported model variant: {args.variant_name}. Available: {list(MODEL_BUILDERS.keys())}")

        with strategy.scope():
            model = model_builder(input_size=(
                args.img_height, args.img_width, args.input_channels))
            model.compile(optimizer='adam', loss='binary_crossentropy')
            print(f"Loading weights from {args.weights_path}...")
            model.load_weights(args.weights_path)
            print("Weights loaded successfully.")
            model.summary(line_length=120)
    except Exception as e:
        print(f"Error building model or loading weights: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Run Prediction and Measure Time
    print("\nRunning inference and measuring time...")
    all_predictions = []
    total_prediction_time = 0.0
    num_batches = 0
    inference_iterator = iter(inference_dataset)
    start_inference_time = time.time()

    file_to_batch_map = {}
    batch_idx = 0
    remaining_files = h5_files.copy()

    patient_batch_times = defaultdict(list)
    batch_patient_map = {}

    for batch_data in inference_iterator:
        batch_files = remaining_files[:global_batch_size]
        remaining_files = remaining_files[global_batch_size:]

        batch_patients = set()
        for i, file_path in enumerate(batch_files):
            patient_id = extract_patient_id(file_path)
            batch_patients.add(patient_id)
            file_to_batch_map[file_path] = (batch_idx, i)
        batch_patient_map[batch_idx] = list(batch_patients)

        images, _ = batch_data
        batch_start_time = time.time()
        batch_predictions = model.predict_on_batch(images)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_prediction_time += batch_time

        for patient_id in batch_patients:
            patient_batch_times[patient_id].append(batch_time)

        all_predictions.append(batch_predictions)
        num_batches += 1
        print(f"Processed batch {num_batches}...", end='\r')
        batch_idx += 1

    end_inference_time = time.time()
    total_wall_time = end_inference_time - start_inference_time
    print(f"\nProcessed {num_batches} batches.")

    patient_total_times = {}
    patient_avg_slice_time = {}
    for patient_id, batch_times in patient_batch_times.items():
        patient_total_times[patient_id] = sum(batch_times)
        num_slices = len(patient_slices[patient_id])
        patient_avg_slice_time[patient_id] = patient_total_times[patient_id] / \
            num_slices if num_slices > 0 else 0

    avg_patient_time = sum(patient_total_times.values()) / \
        len(patient_total_times) if patient_total_times else 0

    timing_results = {
        'total_prediction_time_s': total_prediction_time,
        'total_wall_time_s': total_wall_time,
        'num_batches': num_batches,
        'avg_time_per_batch_s': total_prediction_time / num_batches if num_batches > 0 else 0,
        'avg_time_per_slice_s': total_prediction_time / num_inference_slices if num_inference_slices > 0 else 0,
        'total_slices': num_inference_slices,
        'global_batch_size': global_batch_size,
        'num_patients': num_patients,
        'avg_time_per_patient_s': avg_patient_time,
        'patient_times': patient_total_times,
        'patient_avg_slice_times': patient_avg_slice_time,
        'top_5_patients_by_time': sorted(patient_total_times.items(), key=lambda x: x[1], reverse=True)[:5]
    }

    print("\n--- Inference Timing Results ---")
    if num_batches > 0:
        avg_time_per_batch = timing_results['avg_time_per_batch_s']
        print(
            f"Total prediction time (sum of predict_on_batch): {total_prediction_time:.4f} seconds")
        print(
            f"Total wall clock time for prediction loop: {total_wall_time:.4f} seconds")
        print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
        if num_inference_slices > 0:
            avg_time_per_slice = timing_results['avg_time_per_slice_s']
            print(f"Average time per slice: {avg_time_per_slice:.6f} seconds")
        else:
            print("Average time per slice: N/A (slice count unknown)")
        print(f"Total slices processed: {num_inference_slices}")
        print(f"Global batch size used: {global_batch_size}")
        print(f"Number of patients: {num_patients}")
        print(f"Average time per patient: {avg_patient_time:.4f} seconds")
    else:
        print("Warning: No batches processed during inference.")

    timing_report_path = os.path.join(
        output_dir, f"{args.variant_name}_inference_timing_report.txt")  # Use constructed path
    try:
        with open(timing_report_path, 'w') as f:
            f.write(f"Inference Timing Report for {args.variant_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {args.variant_name}\n")
            f.write(f"Weights: {args.weights_path}\n")
            f.write(
                f"Input Size: {args.img_height}x{args.img_width}x{args.input_channels}\n")
            f.write(
                f"Batch Size (per replica): {args.batch_size_per_replica}\n")
            f.write(f"Global Batch Size: {global_batch_size}\n")
            f.write(f"Mixed Precision: {args.use_mixed_precision}\n\n")

            f.write("General Timing:\n")
            f.write("--------------\n")
            f.write(
                f"Total prediction time: {timing_results['total_prediction_time_s']:.4f} seconds\n")
            f.write(
                f"Total wall clock time: {timing_results['total_wall_time_s']:.4f} seconds\n")
            f.write(f"Number of batches: {timing_results['num_batches']}\n")
            f.write(
                f"Average time per batch: {timing_results['avg_time_per_batch_s']:.4f} seconds\n")
            f.write(
                f"Total slices processed: {timing_results['total_slices']}\n")
            f.write(
                f"Average time per slice: {timing_results['avg_time_per_slice_s']:.6f} seconds\n\n")

            f.write("Patient-Level Statistics:\n")
            f.write("-----------------------\n")
            f.write(f"Number of patients: {timing_results['num_patients']}\n")
            f.write(
                f"Average time per patient: {timing_results['avg_time_per_patient_s']:.4f} seconds\n\n")

            f.write("Top 5 Patients by Processing Time:\n")
            for i, (patient_id, time_taken) in enumerate(timing_results['top_5_patients_by_time'], 1):
                slices = len(patient_slices[patient_id])
                f.write(
                    f"{i}. {patient_id}: {time_taken:.4f} seconds ({slices} slices, {time_taken/slices:.6f} s/slice)\n")

            f.write(
                "\nNote: Patient timing is approximate as batches may contain slices from multiple patients.\n")

        print(f"\nTiming report saved to: {timing_report_path}")
    except Exception as e:
        print(f"Error saving timing report: {e}")

    # 5. Save Prediction Examples (Visual Overlays)
    if args.num_examples_to_save > 0:
        print(
            f"\nSaving {args.num_examples_to_save} visual prediction examples...")
        save_prediction_examples(
            model=model,
            dataset=inference_dataset,
            output_folder=output_dir,  # Use constructed path
            num_examples=args.num_examples_to_save,
            threshold=args.threshold
        )
    else:
        print("\nSkipping saving visual prediction examples.")

    print("\n--- Inference Finished ---")
    total_script_time = time.time() - script_start_time
    print(f"Total script execution time: {total_script_time:.2f} seconds")
    print(f"Full timing details saved to: {timing_report_path}")
    final_cleanup(model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Inference using a trained AS-Net model.")

    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained model weights file (.weights.h5 or similar).')
    parser.add_argument('--h5_data_dir', type=str, required=True,
                        help='Path to the directory containing preprocessed H5 slices for inference.')
    parser.add_argument('--output_subdir', type=str, default='results',
                        help='Subdirectory name within ./BraTS23_inference_output/<variant_name>/ to save results.')
    parser.add_argument('--variant_name', type=str, required=True,
                        choices=MODEL_BUILDERS.keys(),
                        help=f'Name of the model variant used for training (e.g., {list(MODEL_BUILDERS.keys())}).')

    parser.add_argument('--img_height', type=int,
                        default=224, help='Target image height.')
    parser.add_argument('--img_width', type=int,
                        default=224, help='Target image width.')
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels in H5 files.')
    parser.add_argument('--batch_size_per_replica', type=int,
                        default=32, help='Batch size per GPU replica.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for generating binary predictions.')
    parser.add_argument('--num_examples_to_save', type=int, default=5,
                        help='Number of visual comparison examples to save.')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision (float16) computation.')

    args = parser.parse_args()

    run_inference(args)
