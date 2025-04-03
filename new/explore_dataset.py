# explore_dataset.py

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.util import montage
import random  # For sampling
import traceback  # For error details

# --- Configuration ---
DATASET_PATH = "brats2020-training-data/"  # Path to the dataset base directory
METADATA_FILE = os.path.join(DATASET_PATH, "BraTS20 Training Metadata.csv")
H5_DATA_DIR = os.path.join(
    DATASET_PATH, "BraTS2020_training_data/content/data")
# Directory to save visualizations
EXPLORE_OUTPUT_DIR = "dataset_exploration_output"

# --- Create Output Directory ---
os.makedirs(EXPLORE_OUTPUT_DIR, exist_ok=True)
print(f"Dataset exploration outputs will be saved to: {EXPLORE_OUTPUT_DIR}")

# --- Helper Functions ---


def load_h5_slice(file_path):
    """Load a single H5 file containing one slice with all modalities"""
    with h5py.File(file_path, 'r') as hf:
        image_data = hf['image'][()]  # (H, W, 4)
        mask_data = hf['mask'][()]   # (H, W, 3)
    return image_data, mask_data


def get_modality_names():
    """Return the names of the four modalities in the dataset"""
    return ['T1', 'T1ce', 'T2', 'FLAIR']


def get_tumor_region_names():
    """Return the names of the tumor regions in the dataset"""
    return ['NCR/NET', 'ED', 'ET']  # Shortened names for plots


def create_colormap_for_segments():
    """Create a colormap for the segmentation visualization"""
    # Colors from original script (Viridis-like)
    colors = ['#440054', '#3b528b', '#18b880',
              '#e6d74f']  # Background, NCR/NET, ED, ET
    return mcolors.ListedColormap(colors)


def get_patient_slices(metadata_df, patient_id):
    """Get all valid slice file paths for a specific patient, sorted by slice index."""
    if 'patient_id' not in metadata_df.columns:
        print("Error: 'patient_id' column not in metadata")
        return []

    patient_df = metadata_df[metadata_df['patient_id'] == patient_id].copy()
    if 'slice' in patient_df.columns:
        patient_df = patient_df.sort_values('slice')
    else:
        print("Warning: 'slice' column not found for sorting.")

    slice_paths = []
    for _, row in patient_df.iterrows():
        if 'slice_path' in row and isinstance(row['slice_path'], str):
            base_filename = os.path.basename(row['slice_path'])
            h5_path = os.path.join(H5_DATA_DIR, base_filename)
            if os.path.exists(h5_path):
                slice_paths.append(h5_path)
            # else: print(f"Warning: Slice file not found: {h5_path}") # Optional warning
    print(f"Found {len(slice_paths)} valid slices for patient {patient_id}")
    return slice_paths


def reconstruct_3d_volume(slice_paths, modality_index=1):
    """Reconstruct a 3D volume from 2D slices for a specific modality"""
    if not slice_paths:
        print("No slice paths provided")
        return None

    try:
        img_0, _ = load_h5_slice(slice_paths[0])
        height, width = img_0.shape[:2]
        num_slices = len(slice_paths)
        volume = np.zeros((height, width, num_slices), dtype=img_0.dtype)

        for i, path in enumerate(slice_paths):
            try:
                img, _ = load_h5_slice(path)
                if img.shape[0] == height and img.shape[1] == width:
                    volume[:, :, i] = img[:, :, modality_index]
                else:
                    print(
                        f"Warning: Skipping slice {i} due to shape mismatch ({img.shape[:2]} vs expected {(height, width)})")
            except Exception as e:
                print(
                    f"Error loading slice {i} from {os.path.basename(path)}: {e}")
        return volume
    except Exception as e:
        print(
            f"Error initializing volume from first slice {os.path.basename(slice_paths[0])}: {e}")
        return None

# --- Visualization Functions ---


def visualize_sample_slice(file_path, output_dir):
    """Visualize all modalities and segmentation for a single slice and save the plot."""
    try:
        image_data, mask_data = load_h5_slice(file_path)
        modality_names = get_modality_names()
        tumor_region_names = get_tumor_region_names()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()  # Easier indexing

        # Plot modalities
        for i in range(4):
            ax = axes[i]
            ax.imshow(image_data[:, :, i], cmap='gray')
            ax.set_title(modality_names[i])
            ax.axis('off')

        # Create combined segmentation mask (0: Background, 1: NCR/NET, 2: ED, 3: ET)
        segmentation = np.zeros(mask_data.shape[:2], dtype=int)
        for i in range(mask_data.shape[2]):
            segmentation[mask_data[:, :, i] > 0] = i + 1  # Assign 1, 2, 3

        # Plot combined segmentation
        cmap = create_colormap_for_segments()
        # Boundaries for 0, 1, 2, 3
        norm = mcolors.BoundaryNorm(np.arange(0, 5) - 0.5, cmap.N)

        seg_ax = axes[4]  # Put segmentation in the 5th slot
        seg_ax.imshow(image_data[:, :, 1], cmap='gray',
                      alpha=0.7)  # T1ce background
        seg_ax.imshow(segmentation, cmap=cmap, norm=norm,
                      alpha=0.5 * (segmentation > 0))  # Overlay non-background
        seg_ax.set_title('Combined Segmentation')
        seg_ax.axis('off')

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap.colors[0], edgecolor='w', label='Background'),
            Patch(facecolor=cmap.colors[1], edgecolor='w',
                  label=tumor_region_names[0]),  # NCR/NET
            Patch(facecolor=cmap.colors[2], edgecolor='w',
                  label=tumor_region_names[1]),  # ED
            Patch(facecolor=cmap.colors[3], edgecolor='w',
                  label=tumor_region_names[2])  # ET
        ]

        axes[5].axis('off')  # Use the 6th slot for the legend
        axes[5].legend(handles=legend_elements, loc='center', title="Regions")

        plt.suptitle(f"Slice: {os.path.basename(file_path)}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

        save_filename = f"sample_slice_{os.path.basename(file_path).replace('.h5', '.png')}"
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved sample slice visualization to: {save_path}")

    except Exception as e:
        print(f"Error visualizing slice {os.path.basename(file_path)}: {e}")
        traceback.print_exc()


def show_3d_views(volume, modality_name, patient_id, output_dir):
    """Show transverse, frontal, and sagittal views of a 3D volume and save the plot."""
    if volume is None or volume.ndim != 3 or 0 in volume.shape:
        print(f"  Invalid volume for {modality_name}, cannot display views.")
        return

    middle_z = volume.shape[2] // 2
    middle_y = volume.shape[0] // 2
    middle_x = volume.shape[1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Transverse view (axial) - z-slice
    axes[0].imshow(volume[:, :, middle_z], cmap='gray')
    axes[0].set_title(f'Transverse (Axial) - Slice {middle_z}')
    axes[0].axis('off')

    # Frontal view (coronal) - y-slice
    # Transpose for correct orientation
    coronal_slice = volume[middle_y, :, :].T
    axes[1].imshow(coronal_slice, cmap='gray',
                   aspect='auto')  # Adjust aspect ratio
    axes[1].set_title(f'Frontal (Coronal) - Slice {middle_y}')
    axes[1].axis('off')

    # Sagittal view - x-slice
    sagittal_slice = volume[:, middle_x, :].T  # Transpose
    axes[2].imshow(sagittal_slice, cmap='gray', aspect='auto')
    axes[2].set_title(f'Sagittal - Slice {middle_x}')
    axes[2].axis('off')

    plt.suptitle(
        f"Patient {patient_id} - {modality_name} - 3D Views (Middle Slices)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    save_filename = f"patient_{patient_id}_{modality_name}_3d_views.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 3D views visualization to: {save_path}")


def visualize_segmentation_masks(file_path, output_dir):
    """Visualize all segmentation masks individually and save the plot."""
    try:
        image_data, mask_data = load_h5_slice(file_path)
        modality_names = get_modality_names()
        tumor_region_names = get_tumor_region_names()
        background_modality_idx = 1  # T1ce as background

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Original modality
        axes[0].imshow(image_data[:, :, background_modality_idx], cmap='gray')
        axes[0].set_title(
            f'Background ({modality_names[background_modality_idx]})')
        axes[0].axis('off')

        # Combined mask
        segmentation = np.zeros(mask_data.shape[:2], dtype=int)
        for i in range(mask_data.shape[2]):
            segmentation[mask_data[:, :, i] > 0] = i + 1
        cmap_segments = create_colormap_for_segments()
        norm = mcolors.BoundaryNorm(np.arange(0, 5) - 0.5, cmap_segments.N)

        axes[1].imshow(image_data[:, :, background_modality_idx],
                       cmap='gray', alpha=0.7)
        axes[1].imshow(segmentation, cmap=cmap_segments,
                       norm=norm, alpha=0.5 * (segmentation > 0))
        axes[1].set_title('Combined Regions Overlay')
        axes[1].axis('off')

        # Individual masks overlay
        for i in range(mask_data.shape[2]):
            ax = axes[i + 2]
            ax.imshow(
                image_data[:, :, background_modality_idx], cmap='gray', alpha=0.7)
            mask = mask_data[:, :, i]
            # Use a single color map like Reds
            ax.imshow(mask, cmap='Reds', alpha=0.6 * (mask > 0))
            ax.set_title(f'Region: {tumor_region_names[i]}')
            ax.axis('off')

        axes[5].axis('off')  # Hide unused axis if only 3 regions

        plt.suptitle(f"Segmentation Masks: {os.path.basename(file_path)}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_filename = f"segmentation_masks_{os.path.basename(file_path).replace('.h5', '.png')}"
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved segmentation masks visualization to: {save_path}")

    except Exception as e:
        print(
            f"Error visualizing segmentation masks for {os.path.basename(file_path)}: {e}")
        traceback.print_exc()


def create_montage_of_slices(patient_slices, modality_index, modality_name, patient_id, output_dir, skip_ends=True, max_slices=36):
    """Create a montage of slices for a given patient and modality and save the plot."""
    if not patient_slices:
        print("  No slices to display for montage.")
        return

    effective_slices = patient_slices
    if skip_ends and len(patient_slices) > 100:
        print(
            f"  Skipping first/last 50 slices for montage. Showing {len(patient_slices) - 100} middle slices.")
        effective_slices = patient_slices[50:-50]
    elif skip_ends:
        print("  Not enough slices to skip ends for montage.")

    if not effective_slices:
        print("  No slices remain after skipping ends.")
        return

    slices_data = []
    for path in effective_slices:
        try:
            img, _ = load_h5_slice(path)
            slices_data.append(img[:, :, modality_index])
        except Exception as e:
            print(f"  Error loading {os.path.basename(path)} for montage: {e}")

    if not slices_data:
        print("  Failed to load any slices for montage.")
        return

    # Subsample if too many slices
    if len(slices_data) > max_slices:
        step = max(1, len(slices_data) // max_slices)
        slices_data = slices_data[::step]
        print(f"  Subsampled to {len(slices_data)} slices for montage.")

    try:
        montage_arr = montage(np.array(slices_data), padding_width=1, fill=np.min(
            slices_data))  # Add padding
    except Exception as e:
        print(f"  Error creating montage: {e}")
        return

    plt.figure(figsize=(15, 15))
    plt.imshow(montage_arr, cmap='gray')
    plt.title(
        f'Patient {patient_id} - {modality_name} Montage ({len(slices_data)} slices)')
    plt.axis('off')

    save_filename = f"patient_{patient_id}_{modality_name}_montage.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved montage to: {save_path}")


def visualize_modalities_for_different_patients(metadata_df, output_dir, num_patients=3):
    """Visualize modalities and segmentation for different patients and save plots."""
    if 'patient_id' not in metadata_df.columns:
        print("Error: 'patient_id' column not in metadata")
        return

    unique_patients = metadata_df['patient_id'].unique()
    if len(unique_patients) < num_patients:
        num_patients = len(unique_patients)
        print(f"Warning: Only {num_patients} unique patients found.")

    selected_patients = unique_patients[:num_patients]

    for patient_id in selected_patients:
        print(f"\nProcessing Patient: {patient_id}")
        patient_slices = get_patient_slices(metadata_df, patient_id)

        if not patient_slices:
            print(f"  No slices found for patient {patient_id}")
            continue

        # Find a representative slice (e.g., middle or one with tumor)
        slice_to_show_path = patient_slices[len(
            patient_slices) // 2]  # Default middle
        # Search middle third
        for path in patient_slices[len(patient_slices)//3: 2*len(patient_slices)//3]:
            try:
                _, mask = load_h5_slice(path)
                if np.any(mask > 0):
                    slice_to_show_path = path
                    print(
                        f"  Found slice with tumor: {os.path.basename(slice_to_show_path)}")
                    break
            except:
                continue

        try:
            image_data, mask_data = load_h5_slice(slice_to_show_path)
            modality_names = get_modality_names()
            tumor_region_names = get_tumor_region_names()

            fig, axes = plt.subplots(1, 5, figsize=(20, 5))

            # Modalities
            for i, mod_name in enumerate(modality_names):
                axes[i].imshow(image_data[:, :, i], cmap='gray')
                axes[i].set_title(mod_name)
                axes[i].axis('off')

            # Segmentation
            segmentation = np.zeros(mask_data.shape[:2], dtype=int)
            for i in range(mask_data.shape[2]):
                segmentation[mask_data[:, :, i] > 0] = i + 1
            cmap = create_colormap_for_segments()
            norm = mcolors.BoundaryNorm(np.arange(0, 5) - 0.5, cmap.N)

            axes[4].imshow(image_data[:, :, 1], cmap='gray',
                           alpha=0.7)  # T1ce background
            axes[4].imshow(segmentation, cmap=cmap, norm=norm,
                           alpha=0.5 * (segmentation > 0))
            axes[4].set_title('Segmentation')
            axes[4].axis('off')

            plt.suptitle(
                f"Patient {patient_id} - Slice {os.path.basename(slice_to_show_path).replace('.h5','')}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])

            save_filename = f"patient_{patient_id}_modalities_comparison.png"
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved modalities comparison to: {save_path}")

        except Exception as e:
            print(
                f"Error visualizing modalities for patient {patient_id}, slice {os.path.basename(slice_to_show_path)}: {e}")
            traceback.print_exc()

# --- Analysis Functions ---


def analyze_tumor_distribution(patient_slices, patient_id, output_dir):
    """Analyze the distribution of tumor regions for a patient and save plot."""
    if not patient_slices:
        print("  No slices to analyze tumor distribution.")
        return None

    total_pixels = 0
    tumor_region_pixels = [0, 0, 0]  # NCR/NET, ED, ET

    for path in patient_slices:
        try:
            _, mask = load_h5_slice(path)
            if total_pixels == 0:  # Get dimensions from first valid slice
                # Approx total pixels in volume
                total_pixels = mask.shape[0] * \
                    mask.shape[1] * len(patient_slices)

            for i in range(mask.shape[2]):
                tumor_region_pixels[i] += np.sum(mask[:, :, i] > 0)
        except Exception as e:
            print(
                f"  Error processing {os.path.basename(path)} for tumor analysis: {e}")

    if total_pixels == 0:
        print("Could not calculate total pixels.")
        return None

    tumor_region_names = get_tumor_region_names()
    percentages = [(pixels / total_pixels) * 100 if total_pixels >
                   0 else 0 for pixels in tumor_region_pixels]

    data = {
        'Tumor Region': tumor_region_names,
        'Pixel Count': tumor_region_pixels,
        'Percentage (%)': percentages
    }
    distribution_df = pd.DataFrame(data)

    print("\n  Tumor Region Distribution:")
    print(distribution_df)

    # Plot distribution
    plt.figure(figsize=(8, 5))
    plt.bar(tumor_region_names, percentages, color=[
            '#3b528b', '#18b880', '#e6d74f'])
    plt.ylabel('Percentage of Total Volume Pixels (%)')
    plt.title(f'Patient {patient_id} - Tumor Region Distribution')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    save_filename = f"patient_{patient_id}_tumor_distribution.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved tumor distribution plot to: {save_path}")

    return distribution_df


def analyze_class_distribution(h5_data_dir, output_dir, sample_size=100):
    """Analyze distribution of tumor classes across a sample of the dataset and save plot."""
    if not os.path.exists(h5_data_dir):
        print(f"Error: H5 data directory not found: {h5_data_dir}")
        return None

    h5_files = [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')]
    if not h5_files:
        print("No H5 files found in the directory.")
        return None

    if len(h5_files) > sample_size:
        sample_files = random.sample(h5_files, sample_size)
        print(
            f"Analyzing class distribution using a random sample of {sample_size} slices.")
    else:
        sample_files = h5_files
        print(
            f"Analyzing class distribution using all {len(h5_files)} available slices.")

    total_pixels = 0
    background_pixels = 0
    tumor_region_pixels = [0, 0, 0]  # NCR/NET, ED, ET

    for file_name in sample_files:
        try:
            file_path = os.path.join(h5_data_dir, file_name)
            _, mask = load_h5_slice(file_path)

            slice_total = mask.shape[0] * mask.shape[1]
            total_pixels += slice_total

            all_tumor = np.zeros(mask.shape[:2], dtype=bool)
            for i in range(mask.shape[2]):
                tumor_pixels = mask[:, :, i] > 0
                all_tumor |= tumor_pixels
                tumor_region_pixels[i] += np.sum(tumor_pixels)
            background_pixels += slice_total - np.sum(all_tumor)

        except Exception as e:
            print(f"Error processing {file_name} for class analysis: {e}")

    if total_pixels == 0:
        print("Could not process any slices for class distribution.")
        return None

    region_names = ['Background'] + get_tumor_region_names()
    pixel_counts = [background_pixels] + tumor_region_pixels
    percentages = [(count / total_pixels) * 100 for count in pixel_counts]

    data = {'Region': region_names, 'Pixel Count': pixel_counts,
            'Percentage (%)': percentages}
    distribution_df = pd.DataFrame(data)

    print("\nOverall Class Distribution (from sample):")
    print(distribution_df)

    # Plot distribution
    plt.figure(figsize=(8, 5))
    colors = ['#440054', '#3b528b', '#18b880',
              '#e6d74f']  # Background, NCR/NET, ED, ET
    plt.bar(region_names, percentages, color=colors)
    plt.ylabel('Percentage of Total Pixels (%)')
    plt.title(f'Overall Class Distribution (Sample Size: {len(sample_files)})')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    save_filename = "overall_class_distribution.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overall class distribution plot to: {save_path}")

    return distribution_df

# --- Summary Function ---


def summarize_dataset(metadata_df, h5_data_dir):
    """Provide a summary of the dataset properties"""
    print("\n" + "="*60)
    print(" " * 15 + "BraTS 2020 H5 Dataset Summary")
    print("="*60)

    # Metadata summary
    if metadata_df is not None and not metadata_df.empty:
        print("\n--- Metadata Summary ---")
        print(f"Total metadata entries (slices): {len(metadata_df)}")
        if 'patient_id' in metadata_df.columns:
            print(
                f"Unique patients found in metadata: {metadata_df['patient_id'].nunique()}")
        if 'slice' in metadata_df.columns:
            print(
                f"Slice index range in metadata: {metadata_df['slice'].min()} - {metadata_df['slice'].max()}")
        else:
            print("Slice index column ('slice') not found.")
    else:
        print("\nMetadata not loaded or empty.")

    # File summary
    print("\n--- File System Summary ---")
    if os.path.exists(h5_data_dir):
        h5_files = [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')]
        print(f"Total H5 files found in directory: {len(h5_files)}")
        if h5_files:
            # Check one file for dimensions/structure
            example_file = os.path.join(h5_data_dir, h5_files[0])
            try:
                with h5py.File(example_file, 'r') as hf:
                    print("\nExample H5 file structure:")
                    print(f"  Keys: {list(hf.keys())}")
                    if 'image' in hf:
                        print(
                            f"  Image dimensions: {hf['image'].shape} (H, W, Modalities)")
                    if 'mask' in hf:
                        print(
                            f"  Mask dimensions: {hf['mask'].shape} (H, W, Tumor Regions)")
            except Exception as e:
                print(f"  Error reading example H5 file structure: {e}")
    else:
        print(f"H5 data directory not found: {h5_data_dir}")

    # Content summary
    print("\n--- Content Summary ---")
    print(f"Expected Modalities (Image Channel Order): {get_modality_names()}")
    print(
        f"Expected Tumor Regions (Mask Channel Order): {get_tumor_region_names()}")
    print("="*60 + "\n")


# --- Main Execution Logic ---
if __name__ == "__main__":
    print("Starting BraTS Dataset Exploration...")

    # 1. Load Metadata
    print("\n--- Loading Metadata ---")
    metadata_df = None
    if os.path.exists(METADATA_FILE):
        try:
            metadata_df = pd.read_csv(METADATA_FILE)
            print(f"Loaded metadata with {len(metadata_df)} entries.")
            if 'volume' in metadata_df.columns and 'patient_id' not in metadata_df.columns:
                # Assuming volume number is patient ID
                metadata_df['patient_id'] = metadata_df['volume']
                print("Created 'patient_id' column from 'volume'.")
            if 'patient_id' in metadata_df.columns:
                print(
                    f"Unique patients in metadata: {metadata_df['patient_id'].nunique()}")
            else:
                print("Warning: Could not identify a 'patient_id' column.")
            # print(metadata_df.head()) # Optional: view head
        except Exception as e:
            print(f"Error loading metadata file {METADATA_FILE}: {e}")
    else:
        print(f"Error: Metadata file not found at {METADATA_FILE}")

    # 2. Check H5 Data Directory
    print("\n--- Checking H5 Data Directory ---")
    h5_files_list = []
    if os.path.exists(H5_DATA_DIR):
        h5_files_list = [f for f in os.listdir(
            H5_DATA_DIR) if f.endswith('.h5')]
        if h5_files_list:
            print(f"Found {len(h5_files_list)} H5 files in {H5_DATA_DIR}.")
        else:
            print(f"Warning: No H5 files found in {H5_DATA_DIR}.")
    else:
        print(f"Error: H5 data directory not found at {H5_DATA_DIR}.")

    # Proceed only if data and metadata seem available
    if metadata_df is not None and not metadata_df.empty and h5_files_list:

        # 3. Visualize a Sample Slice
        print("\n--- Visualizing Sample Slice ---")
        sample_file_path = None
        # Try to find a slice with tumor for better visualization
        for fname in h5_files_list[:50]:  # Check first 50
            fpath = os.path.join(H5_DATA_DIR, fname)
            try:
                _, mask_d = load_h5_slice(fpath)
                if np.any(mask_d > 0):
                    sample_file_path = fpath
                    print(f"Found sample slice with tumor: {fname}")
                    break
            except:
                continue
        if not sample_file_path:  # Fallback to first file
            sample_file_path = os.path.join(H5_DATA_DIR, h5_files_list[0])
            print(f"Using first slice as sample: {h5_files_list[0]}")
        visualize_sample_slice(sample_file_path, EXPLORE_OUTPUT_DIR)
        visualize_segmentation_masks(sample_file_path, EXPLORE_OUTPUT_DIR)

        # 4. Reconstruct 3D Views for one patient
        print("\n--- Reconstructing 3D Views (Example Patient) ---")
        if 'patient_id' in metadata_df.columns:
            # Just take the first patient ID
            example_patient_id = metadata_df['patient_id'].iloc[0]
            print(f"Using Patient ID: {example_patient_id}")
            example_patient_slices = get_patient_slices(
                metadata_df, example_patient_id)
            if example_patient_slices:
                mod_names = get_modality_names()
                for idx, mod_name in enumerate(mod_names):
                    print(f" Reconstructing {mod_name} volume...")
                    vol = reconstruct_3d_volume(
                        example_patient_slices, modality_index=idx)
                    show_3d_views(
                        vol, mod_name, example_patient_id, EXPLORE_OUTPUT_DIR)

                # 5. Analyze Tumor Distribution for the example patient
                print("\n--- Analyzing Tumor Distribution (Example Patient) ---")
                analyze_tumor_distribution(
                    example_patient_slices, example_patient_id, EXPLORE_OUTPUT_DIR)

                # 6. Create Montage for the example patient
                print("\n--- Creating Montages (Example Patient) ---")
                for idx, mod_name in enumerate(mod_names):
                    print(f" Creating {mod_name} montage...")
                    create_montage_of_slices(
                        example_patient_slices, idx, mod_name, example_patient_id, EXPLORE_OUTPUT_DIR)
            else:
                print(
                    f"Could not retrieve slices for patient {example_patient_id}.")
        else:
            print("Cannot perform patient-specific analysis without 'patient_id'.")

        # 7. Visualize Modalities for Different Patients
        print("\n--- Visualizing Modalities Comparison (Multiple Patients) ---")
        visualize_modalities_for_different_patients(
            metadata_df, EXPLORE_OUTPUT_DIR, num_patients=3)

        # 8. Analyze Overall Class Distribution
        print("\n--- Analyzing Overall Class Distribution ---")
        # Increase sample size if desired
        analyze_class_distribution(
            H5_DATA_DIR, EXPLORE_OUTPUT_DIR, sample_size=100)

        # 9. Final Summary
        summarize_dataset(metadata_df, H5_DATA_DIR)

    else:
        print("\nSkipping detailed exploration due to missing metadata or data files.")

    print("\nDataset Exploration Script Finished.")
