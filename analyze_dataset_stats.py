\
import os
import glob
import re
from collections import defaultdict

# --- Configuration ---
H5_BASE_DIR = './preprocessed_brats23_h5_slices'
SPLITS = ['train', 'validation', 'test']
# Regex to extract patient ID (adjust if filename pattern is different)
# Assumes pattern like: BraTS-GLI-XXXXX-XXX_slice_YYY.h5
PATIENT_ID_REGEX = re.compile(r"^(.*?)_slice_\d+\.h5$")
# --- End Configuration ---

def analyze_stats(base_dir):
    """Analyzes the H5 dataset to count slices and patients."""
    total_slices = 0
    patient_ids_by_split = {split: set() for split in SPLITS}
    slices_per_split = {split: 0 for split in SPLITS}
    slices_per_patient = defaultdict(int) # Count slices per patient globally

    print(f"Analyzing dataset in: {base_dir}")

    for split in SPLITS:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: Directory not found for split '{split}': {split_dir}")
            continue

        h5_files = glob.glob(os.path.join(split_dir, "*.h5"))
        num_files = len(h5_files)
        slices_per_split[split] = num_files
        total_slices += num_files

        print(f"Found {num_files} slices in '{split}' split.")

        for f_path in h5_files:
            filename = os.path.basename(f_path)
            match = PATIENT_ID_REGEX.match(filename)
            if match:
                patient_id = match.group(1)
                patient_ids_by_split[split].add(patient_id)
                slices_per_patient[patient_id] += 1
            else:
                print(f"Warning: Could not extract patient ID from filename: {filename}")

    all_patient_ids = set().union(*patient_ids_by_split.values())
    total_unique_patients = len(all_patient_ids)

    print("\\n--- Dataset Statistics ---")
    print(f"Total Slices (across all splits): {total_slices}")
    for split in SPLITS:
        print(f"  - Slices in '{split}': {slices_per_split[split]}")

    print(f"\\nTotal Unique Patients (across all splits): {total_unique_patients}")
    for split in SPLITS:
        print(f"  - Unique Patients in '{split}': {len(patient_ids_by_split[split])}")

    # Optional: Print slice distribution summary
    if slices_per_patient:
        avg_slices = total_slices / total_unique_patients if total_unique_patients else 0
        min_slices = min(slices_per_patient.values())
        max_slices = max(slices_per_patient.values())
        print("\\nSlice Distribution per Patient:")
        print(f"  - Average slices per patient: {avg_slices:.2f}")
        print(f"  - Minimum slices for a patient: {min_slices}")
        print(f"  - Maximum slices for a patient: {max_slices}")
    print("--------------------------\\n")

if __name__ == "__main__":
    analyze_stats(H5_BASE_DIR)

