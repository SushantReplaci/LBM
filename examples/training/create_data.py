import os
import tarfile
import argparse
import random
from tqdm import tqdm
from pathlib import Path

def _write_shards(samples, output_path, shard_size, split_name):
    """Helper to write shards for a specific split."""
    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    num_shards = (len(samples) + shard_size - 1) // shard_size
    print(f"\nWriting {len(samples)} samples to {split_name} split ({num_shards} shards)...")
    
    for shard_idx in range(num_shards):
        shard_samples = samples[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        shard_filename = split_dir / f"shard-{shard_idx:06d}.tar"
        
        print(f"[{split_name}] Writing shard {shard_idx + 1}/{num_shards}: {shard_filename.name}")
        
        with tarfile.open(shard_filename, "w") as tar:
            for sample in tqdm(shard_samples, desc=f"{split_name} Shard {shard_idx}", leave=False):
                key = sample["key"]
                # Add components with their specific naming conventions
                tar.add(sample["original"], arcname=f"{key}.original.jpg")
                tar.add(sample["edited"], arcname=f"{key}.target.png")
                tar.add(sample["mask"], arcname=f"{key}.mask.png")
                tar.add(sample["edge_map"], arcname=f"{key}.edge_map.png")

def create_webdataset_shards(input_dir, output_dir, shard_size=1000, num_images=None, val_split=0.05, seed=42):
    """
    Creates WebDataset (.tar) shards from a dataset with subfolders,
    supporting train/val splitting.
    """
    input_path = Path(input_dir)
    original_dir = input_path / "input_image"
    mask_dir = input_path / "binary_mask"
    edited_dir = input_path / "edited_image"
    edge_map_dir = input_path / "edge_map"

    required_dirs = [original_dir, mask_dir, edited_dir, edge_map_dir]
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing required subdirectories: {', '.join([d.name for d in missing_dirs])} in {input_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all matching files
    original_files = sorted([f for f in original_dir.iterdir() if f.is_file()])
    samples = []
    
    print(f"Scanning for matching quadruplets in {input_dir}...")
    for orig_file in original_files:
        filename = orig_file.name
        mask_file = mask_dir / filename
        edited_file = edited_dir / filename
        edge_map_file = edge_map_dir / filename
        
        if mask_file.exists() and edited_file.exists() and edge_map_file.exists():
            samples.append({
                "original": orig_file,
                "mask": mask_file,
                "edited": edited_file,
                "edge_map": edge_map_file,
                "key": orig_file.stem
            })
    
    print(f"Found {len(samples)} complete quadruplets.")
    
    # Shuffle samples for stable splitting
    random.seed(seed)
    random.shuffle(samples)

    if num_images is not None and num_images < len(samples):
        print(f"Randomly selecting {num_images} images for toy data...")
        samples = samples[:num_images]
    elif num_images is not None:
        print(f"Requested {num_images} images, but only {len(samples)} are available. Using all.")
    
    if not samples:
        print("No samples found. Exiting.")
        return

    # Split into train and validation
    val_count = int(len(samples) * val_split)
    # Ensure at least one validation sample if split > 0 and samples exist
    if val_split > 0 and val_count == 0 and len(samples) > 0:
        val_count = 1
        
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    print(f"Split distribution: {len(train_samples)} train, {len(val_samples)} validation (split ratio: {val_split})")

    # Create shards for both splits
    if train_samples:
        _write_shards(train_samples, output_path, shard_size, "train")
    
    if val_samples:
        _write_shards(val_samples, output_path, shard_size, "val")

    print(f"\nSuccessfully created shards in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards for LBM object removal training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to root dir containing original, mask, edited folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .tar shards.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to randomly select (for toy data).")
    parser.add_argument("--val_split", type=float, default=0.05, help="Fraction of data to use for validation (default: 0.05).")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of images per shard (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    
    args = parser.parse_args()
    
    create_webdataset_shards(
        args.input_dir, 
        args.output_dir, 
        shard_size=args.shard_size,
        num_images=args.num_images,
        val_split=args.val_split,
        seed=args.seed
    )
