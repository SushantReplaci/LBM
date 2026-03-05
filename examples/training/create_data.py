import os
import tarfile
import argparse
import random
from tqdm import tqdm
from pathlib import Path

def create_webdataset_shards(input_dir, output_dir, shard_size=1000, num_images=None):
    """
    Creates WebDataset (.tar) shards from a dataset with subfolders:
    - original/
    - mask/
    - edited/
    """
    input_path = Path(input_dir)
    original_dir = input_path / "original"
    mask_dir = input_path / "mask"
    edited_dir = input_path / "edited"

    if not all(d.exists() for d in [original_dir, mask_dir, edited_dir]):
        raise FileNotFoundError(f"One or more required subdirectories (original, mask, edited) not found in {input_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all matching files
    original_files = sorted([f for f in original_dir.iterdir() if f.is_file()])
    samples = []
    
    print(f"Scanning for matching triplets in {input_dir}...")
    for orig_file in original_files:
        filename = orig_file.name
        mask_file = mask_dir / filename
        edited_file = edited_dir / filename
        
        if mask_file.exists() and edited_file.exists():
            samples.append({
                "original": orig_file,
                "mask": mask_file,
                "edited": edited_file,
                "key": orig_file.stem
            })
    
    print(f"Found {len(samples)} complete triplets.")
    
    if num_images is not None and num_images < len(samples):
        print(f"Randomly selecting {num_images} images for toy data...")
        random.shuffle(samples)
        samples = samples[:num_images]
    elif num_images is not None:
        print(f"Requested {num_images} images, but only {len(samples)} are available. Using all.")
    
    if not samples:
        print("No samples found. Exiting.")
        return

    # Create shards
    num_shards = (len(samples) + shard_size - 1) // shard_size
    
    for shard_idx in range(num_shards):
        shard_samples = samples[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        shard_filename = output_path / f"shard-{shard_idx:06d}.tar"
        
        print(f"Writing shard {shard_idx + 1}/{num_shards}: {shard_filename.name}")
        
        with tarfile.open(shard_filename, "w") as tar:
            for sample in tqdm(shard_samples, desc=f"Shard {shard_idx}"):
                key = sample["key"]
                
                # Add original as original.jpg
                # We rename internal files to match what train_lbm_removal.py expects
                tar.add(sample["original"], arcname=f"{key}.original.jpg")
                
                # Add edited as target.png
                tar.add(sample["edited"], arcname=f"{key}.target.png")
                
                # Add mask as mask.png
                tar.add(sample["mask"], arcname=f"{key}.mask.png")

    print(f"Successfully created {num_shards} shards in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards for LBM object removal training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to root dir containing original, mask, edited folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .tar shards.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to randomly select (for toy data).")
    
    args = parser.parse_args()
    
    create_webdataset_shards(args.input_dir, args.output_dir, args.num_images)
