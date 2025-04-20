import os
import random
import argparse

# --- Configuration ---
# !! IMPORTANT: Set this to the path of the UNZIPPED BraTS 2023 GLI Training Data !!
BRATS23_TRAIN_DATA_DIR = './BraTS23_TrainingData'  #
OUTPUT_DIR = './BraTS23_data_splits'

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VALIDATION_RATIO (e.g., 0.10)

RANDOM_SEED = 42


def create_splits(data_dir, output_dir, train_r, val_r, seed):
    """Lists cases, shuffles, splits, and saves ID lists."""
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    # List all valid case directories (e.g., BraTS-GLI-XXXXX-XXX)
    try:
        case_ids = [d for d in os.listdir(data_dir) if os.path.isdir(
            os.path.join(data_dir, d)) and d.startswith('BraTS-GLI-')]
    except FileNotFoundError:
        print(f"Error: Cannot list contents of directory: {data_dir}")
        return

    if not case_ids:
        print(
            f"Error: No case directories found in {data_dir}. Check the path and contents.")
        return

    num_cases = len(case_ids)
    print(f"Found {num_cases} cases.")

    # Shuffle the case IDs
    random.seed(seed)
    random.shuffle(case_ids)
    print(f"Shuffled cases using seed {seed}.")

    # Calculate split points
    train_end_idx = int(train_r * num_cases)
    val_end_idx = train_end_idx + int(val_r * num_cases)
    # The rest are test cases

    # Create the splits
    train_ids = case_ids[:train_end_idx]
    val_ids = case_ids[train_end_idx:val_end_idx]
    test_ids = case_ids[val_end_idx:]

    print(
        f"Split sizes: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}")
    assert len(train_ids) + len(val_ids) + \
        len(test_ids) == num_cases, "Split sizes don't match total cases!"

    # Save the splits to files
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, 'train_ids.txt'), 'w') as f:
            f.write('\n'.join(train_ids))
        with open(os.path.join(output_dir, 'val_ids.txt'), 'w') as f:
            f.write('\n'.join(val_ids))
        with open(os.path.join(output_dir, 'test_ids.txt'), 'w') as f:
            f.write('\n'.join(test_ids))
        print(f"Split ID lists saved to {output_dir}")
    except IOError as e:
        print(f"Error saving split files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for BraTS data.")
    parser.add_argument('--data_dir', type=str, default=BRATS23_TRAIN_DATA_DIR,
                        help='Path to the BraTS 2023 GLI Training data directory.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save the split ID files.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for shuffling.')
    parser.add_argument('--train_ratio', type=float,
                        default=TRAIN_RATIO, help='Proportion of data for training.')
    parser.add_argument('--val_ratio', type=float, default=VALIDATION_RATIO,
                        help='Proportion of data for validation.')

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: Train ratio + Validation ratio must be less than 1.0 to leave data for testing.")
    else:
        create_splits(args.data_dir, args.output_dir,
                      args.train_ratio, args.val_ratio, args.seed)
