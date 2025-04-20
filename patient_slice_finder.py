import os
import glob
import argparse
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def find_patient_slices(h5_test_dir, patient_id=None, list_patients=False, show_examples=False):
    """
    Finds all slices belonging to a specific patient in the test dataset or lists all patients.

    Args:
        h5_test_dir: Directory containing the H5 test slices
        patient_id: Patient ID to search for (e.g., 'BraTS-GLI-00123-000')
        list_patients: If True, list all unique patient IDs in the dataset
        show_examples: If True and patient_id provided, show example slices

    Returns:
        List of slice filenames for the specified patient, or dictionary of all patients if list_patients=True
    """
    # Make sure the directory exists
    if not os.path.isdir(h5_test_dir):
        raise FileNotFoundError(
            f"Test slices directory not found: {h5_test_dir}")

    # Get all H5 files in the directory
    h5_files = glob.glob(os.path.join(h5_test_dir, "*.h5"))
    if not h5_files:
        raise ValueError(f"No H5 files found in {h5_test_dir}")

    print(f"Found {len(h5_files)} total test slices.")

    # Group files by patient ID
    patient_slices = defaultdict(list)
    for file_path in h5_files:
        file_name = os.path.basename(file_path)
        # BraTS-GLI-XXXXX-XXX_slice_XXX.h5 format
        patient_part = file_name.split("_slice_")[0]
        patient_slices[patient_part].append(file_path)

    print(f"Found {len(patient_slices)} unique patients in the test set.")

    # List all patients if requested
    if list_patients:
        patient_counts = [(p_id, len(slices))
                          for p_id, slices in patient_slices.items()]
        patient_counts.sort(key=lambda x: x[0])  # Sort by patient ID

        print("\n=== Patient IDs in Test Set ===")
        for idx, (p_id, count) in enumerate(patient_counts, 1):
            print(f"{idx:3d}. {p_id:20s} - {count:3d} slices")

        # Return dictionary of patient IDs and their slice counts
        return {p_id: len(slices) for p_id, slices in patient_slices.items()}

    # If a specific patient ID is requested
    if patient_id:
        if patient_id not in patient_slices:
            print(f"Patient ID '{patient_id}' not found in test set.")
            # Find similar patient IDs (case insensitive partial match)
            similar_ids = [pid for pid in patient_slices.keys()
                           if patient_id.lower() in pid.lower()]
            if similar_ids:
                print("Did you mean one of these patients?")
                for sid in similar_ids:
                    print(f"- {sid} ({len(patient_slices[sid])} slices)")
            return []

        # Get slices for the requested patient
        patient_slice_files = sorted(patient_slices[patient_id])
        print(
            f"Found {len(patient_slice_files)} slices for patient {patient_id}:")

        # Print paths to all slices
        for i, slice_path in enumerate(patient_slice_files, 1):
            slice_name = os.path.basename(slice_path)
            print(f"{i:3d}. {slice_name}")

        # Show examples if requested
        if show_examples and patient_slice_files:
            # Show first, middle, and last slice as examples
            example_indices = [0, len(patient_slice_files)//2, -1]

            plt.figure(figsize=(15, 5))
            for i, idx in enumerate(example_indices):
                if idx < len(patient_slice_files):
                    try:
                        with h5py.File(patient_slice_files[idx], 'r') as f:
                            # Get FLAIR channel (index 1) from the image
                            img = f['image'][()][..., 1]
                            mask = f['mask'][()].squeeze()

                            # Normalize image for display
                            img = (img - np.min(img)) / \
                                (np.max(img) - np.min(img) + 1e-8)

                            # Plot image and overlay
                            plt.subplot(1, 3, i+1)
                            plt.imshow(img, cmap='gray')
                            plt.imshow(mask, alpha=0.3, cmap='hot')
                            plt.title(
                                f"Slice {idx+1}/{len(patient_slice_files)}")
                            plt.axis('off')
                    except Exception as e:
                        print(
                            f"Error displaying slice {patient_slice_files[idx]}: {e}")
            plt.suptitle(f"Example Slices for Patient {patient_id}")
            plt.tight_layout()
            plt.show()

        return patient_slice_files

    print("Please specify a patient_id or set list_patients=True.")
    return []


def export_patient_data(patient_slices, output_csv=None):
    """Exports patient slice information to a CSV file"""
    if not patient_slices:
        print("No patient data to export.")
        return

    # Create a DataFrame
    if isinstance(patient_slices, dict):
        # For the list_patients=True case
        df = pd.DataFrame([
            {"patient_id": patient, "slice_count": count}
            for patient, count in patient_slices.items()
        ])
    else:
        # For the specific patient case
        df = pd.DataFrame([
            {"slice_file": os.path.basename(slice_path)}
            for slice_path in patient_slices
        ])

    # Save to CSV if filename provided, otherwise print
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Data exported to {output_csv}")
    else:
        print("\nData Summary:")
        print(df.head(10))
        if len(df) > 10:
            print(f"...and {len(df)-10} more rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find slices for specific patients in BraTS test dataset")
    parser.add_argument("--test_dir", type=str, default="./BraTS23_preprocessed_h5_slices/test",
                        help="Directory containing test slices in H5 format")
    parser.add_argument("--patient_id", type=str, default=None,
                        help="Specific patient ID to find slices for")
    parser.add_argument("--list_patients", action="store_true",
                        help="List all patients in the test dataset")
    parser.add_argument("--show_examples", action="store_true",
                        help="Show example slice visualizations for the patient")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV file to save results")

    args = parser.parse_args()

    result = find_patient_slices(
        args.test_dir,
        args.patient_id,
        args.list_patients,
        args.show_examples
    )

    if args.output_csv:
        export_patient_data(result, args.output_csv)
