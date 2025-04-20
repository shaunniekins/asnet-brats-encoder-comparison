# preprocess_nifti_to_h5.py
import os
import sys
import numpy as np
import nibabel as nib
import h5py
import argparse
from tqdm import tqdm

# --- Configuration ---
DEFAULT_NIFTI_DIR = './BraTS23_TrainingData'
DEFAULT_SPLIT_DIR = './BraTS23_data_splits'
DEFAULT_OUTPUT_H5_DIR = './BraTS23_preprocessed_h5_slices'

# Preprocessing Parameters (MUST MATCH what the model expects before resizing)
# Indices for T1ce (0) and FLAIR (2) in [t1c, t1n, t2f, t2w]
MODALITY_INDICES_NIFTI = [0, 2]
# Maps loaded [T1ce, FLAIR] -> [T1ce, FLAIR, T1ce]
RGB_MAPPING_INDICES = [0, 1, 0]


# --- Helper Functions ---
def zscore_normalize(image_slice):
    """Applies Z-score normalization to a 2D image slice."""
    mean = np.mean(image_slice)
    std = np.std(image_slice)
    if std == 0:  # Avoid division by zero for blank slices
        return image_slice - mean
    return (image_slice - mean) / (std + 1e-8)  # Add epsilon for safety


def load_ids(filepath):
    """Loads case IDs from a text file."""
    try:
        with open(filepath, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            print(
                f"Warning: ID file is empty or contains only whitespace: {filepath}")
            return None
        return ids
    except FileNotFoundError:
        print(f"Error: ID file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading ID file {filepath}: {e}")
        return None


def process_case(case_id, nifti_base_dir, output_h5_dir):
    """Loads NIfTI, processes each slice, and saves as individual H5 files."""
    case_folder_path = os.path.join(nifti_base_dir, case_id)
    slice_save_count = 0
    try:
        t1c_path = os.path.join(case_folder_path, f"{case_id}-t1c.nii.gz")
        t2f_path = os.path.join(
            case_folder_path, f"{case_id}-t2f.nii.gz")  # FLAIR
        seg_path = os.path.join(
            case_folder_path, f"{case_id}-seg.nii.gz")  # Ground Truth Mask

        required_files = [t1c_path, t2f_path, seg_path]
        if not all(os.path.exists(p) for p in required_files):
            print(
                f"Warning: Skipping case {case_id} due to missing required files (T1c, T2f, or Seg).")
            return 0

        # Load NIfTI volumes
        img_t1c_vol = nib.load(t1c_path).get_fdata(dtype=np.float32)
        img_t2f_vol = nib.load(t2f_path).get_fdata(dtype=np.float32)
        mask_vol_float = nib.load(seg_path).get_fdata()
        # Cast to uint8 for label processing
        mask_vol = mask_vol_float.astype(np.uint8)

        # Simple check for orientation (assuming Z is the last axis for BraTS)
        if not (img_t1c_vol.shape == img_t2f_vol.shape == mask_vol.shape):
            print(f"Warning: Shape mismatch in {case_id}. Skipping. "
                  f"T1c: {img_t1c_vol.shape}, T2f: {img_t2f_vol.shape}, Seg: {mask_vol.shape}")
            return 0

        num_slices = img_t1c_vol.shape[2]  # Assuming Z is the last axis

        for slice_idx in range(num_slices):
            # --- Extract Slice Data ---
            t1c_slice = img_t1c_vol[:, :, slice_idx]
            t2f_slice = img_t2f_vol[:, :, slice_idx]
            seg_slice = mask_vol[:, :, slice_idx]

            # --- Process Image for Model Input ---
            # Select modalities based on MODALITY_INDICES_NIFTI (T1ce, FLAIR)
            modalities_selected = [t1c_slice, t2f_slice]

            # Normalize selected modalities
            normalized_modalities = [zscore_normalize(
                mod) for mod in modalities_selected]

            # Map to 3 Channels using RGB_MAPPING_INDICES [0, 1, 0] -> [T1ce, FLAIR, T1ce]
            input_channels_list = [normalized_modalities[i]
                                   for i in RGB_MAPPING_INDICES]
            model_input_slice = np.stack(
                input_channels_list, axis=-1).astype(np.float32)  # Shape (H, W, 3)

            # --- Process Mask for Training ---
            # Create Binary Mask (Tumor = 1, Background = 0)
            binary_mask_slice = (seg_slice > 0).astype(
                np.float32)  # Shape (H, W)
            # Add channel dimension
            binary_mask_slice = np.expand_dims(
                binary_mask_slice, axis=-1)  # Shape (H, W, 1)

            # --- Save Processed Slice to H5 ---
            # NOTE: We save the ORIGINAL size (e.g., 240x240). Resizing will happen in the tf.data pipeline during training.
            h5_filename = f"{case_id}_slice_{slice_idx:03d}.h5"
            h5_save_path = os.path.join(output_h5_dir, h5_filename)

            try:
                with h5py.File(h5_save_path, 'w') as hf:
                    hf.create_dataset(
                        'image', data=model_input_slice, compression="gzip")
                    hf.create_dataset(
                        'mask', data=binary_mask_slice, compression="gzip")
                slice_save_count += 1
            except Exception as h5_err:
                print(f"Error saving H5 file {h5_save_path}: {h5_err}")
                # Optionally remove partially created file
                if os.path.exists(h5_save_path):
                    os.remove(h5_save_path)

    except FileNotFoundError as e:
        print(f"Error loading files for case {case_id}: {e}")
    except Exception as e:
        print(f"Unexpected error processing case {case_id}: {e}")
    finally:
        return slice_save_count


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 2023 NIfTI data into H5 slices.")
    parser.add_argument('--nifti_dir', type=str, default=DEFAULT_NIFTI_DIR,
                        help='Path to the BraTS 2023 GLI Training data directory (containing case folders).')
    parser.add_argument('--split_dir', type=str, default=DEFAULT_SPLIT_DIR,
                        help='Directory containing train_ids.txt, val_ids.txt, test_ids.txt.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_H5_DIR,
                        help='Base directory to save the preprocessed H5 slice files.')

    args = parser.parse_args()

    print("--- Starting BraTS 2023 NIfTI Preprocessing ---")
    print(f"NIfTI Source: {args.nifti_dir}")
    print(f"Split Files : {args.split_dir}")
    print(f"H5 Output   : {args.output_dir}")

    # --- Validate Paths ---
    if not os.path.isdir(args.nifti_dir):
        print(f"Error: NIfTI directory not found at '{args.nifti_dir}'")
        sys.exit(1)
    if not os.path.isdir(args.split_dir):
        print(f"Error: Split directory not found at '{args.split_dir}'")
        sys.exit(1)

    # --- Create Output Directories ---
    output_train_dir = os.path.join(args.output_dir, 'train')
    output_val_dir = os.path.join(args.output_dir, 'validation')
    output_test_dir = os.path.join(args.output_dir, 'test')
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # --- Load Split IDs ---
    train_ids = load_ids(os.path.join(args.split_dir, 'train_ids.txt'))
    val_ids = load_ids(os.path.join(args.split_dir, 'val_ids.txt'))
    test_ids = load_ids(os.path.join(args.split_dir, 'test_ids.txt'))

    if train_ids is None or val_ids is None or test_ids is None:
        print("Error: Failed to load one or more split ID files. Exiting.")
        sys.exit(1)

    print(
        f"Loaded {len(train_ids)} train IDs, {len(val_ids)} val IDs, {len(test_ids)} test IDs.")

    splits_to_process = {
        'train': (train_ids, output_train_dir),
        'validation': (val_ids, output_val_dir),
        'test': (test_ids, output_test_dir)
    }

    total_slices_processed = 0

    # --- Process each split ---
    for split_name, (ids, output_dir) in splits_to_process.items():
        print(f"\n--- Processing {split_name} split ({len(ids)} cases) ---")
        split_slices_processed = 0
        if not ids:
            print(f"No IDs found for {split_name} split. Skipping.")
            continue

        for case_id in tqdm(ids, desc=f"Processing {split_name} cases"):
            saved_count = process_case(case_id, args.nifti_dir, output_dir)
            split_slices_processed += saved_count

        print(
            f"--- Finished {split_name} split. Saved {split_slices_processed} slices. ---")
        total_slices_processed += split_slices_processed

    print("\n================================================")
    print(
        f"Preprocessing Complete. Total slices saved: {total_slices_processed}")
    print(f"Preprocessed H5 slices saved to: {args.output_dir}")
    print("================================================")
