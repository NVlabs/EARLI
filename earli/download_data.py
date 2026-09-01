# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
Download VRP benchmark problem instances from NVIDIA Labs olist-vrp-benchmark repository.

This script downloads problem_instances.zip from:
https://github.com/NVlabs/olist-vrp-benchmark/blob/main/problem_instances.zip

The file contains VRP problem instances of various sizes (20-500 nodes) for Rio de Janeiro and São Paulo.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# problem_instances.zip as of olist-vrp-benchmark commit 4f7ec0b (pinned below)
EXPECTED_SHA256 = "2b26dc0b51b9b8c4120f59f0cf9f28cd4f4edefaccda36208b71d87ee12d9d90"


def download_file(url, destination, chunk_size=8192):
    """Download a file with progress bar."""
    print(f"Downloading {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Successfully downloaded to {destination}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def verify_sha256(file_path, expected):
    """Check the file's SHA256 digest against the pinned value."""
    digest = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        print(f"Error: SHA256 mismatch for {file_path}")
        print(f"  expected: {expected}")
        print(f"  actual:   {digest.hexdigest()}")
        print("The file may be corrupted or tampered with; it will not be extracted.")
        return False
    return True


def extract_zip(zip_path, extract_to):
    """Extract zip file to destination directory."""
    print(f"Extracting {zip_path} to {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Reject entries that would land outside extract_to (Zip Slip)
            extract_root = Path(extract_to).resolve()
            for member in zip_ref.namelist():
                target = (extract_root / member).resolve()
                if not target.is_relative_to(extract_root):
                    print(f"Error: zip entry escapes extraction directory: {member}")
                    return False
            zip_ref.extractall(extract_to)
        
        print(f"Successfully extracted to {extract_to}")
        return True
        
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download VRP benchmark problem instances from NVIDIA Labs repository"
    )
    parser.add_argument(
        "--datasets-dir", 
        type=str, 
        default="datasets",
        help="Directory to extract datasets to (default: datasets)"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Delete the downloaded zip file after extraction"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args(argv)
    
    # URLs for the data, pinned to the commit matching EXPECTED_SHA256
    base_url = "https://github.com/NVlabs/olist-vrp-benchmark/raw/4f7ec0b2f33fd013199c982878e491aff5e7baaf"
    zip_filename = "problem_instances.zip"
    zip_url = f"{base_url}/{zip_filename}"
    
    # Create datasets directory if it doesn't exist
    datasets_dir = Path(args.datasets_dir)
    datasets_dir.mkdir(exist_ok=True)
    
    zip_path = datasets_dir / zip_filename
    
    # Check if zip file already exists
    if zip_path.exists() and not args.force:
        print(f"File {zip_path} already exists. Use --force to overwrite.")
        if not args.cleanup:
            sys.exit(1)
    
    # Check if data already extracted
    problem_instances_dir = datasets_dir / "problem_instances"
    if problem_instances_dir.exists() and not args.force:
        print(f"Directory {problem_instances_dir} already exists. Use --force to overwrite.")
        if not args.cleanup:
            sys.exit(1)
    
    # Download the zip file
    if not zip_path.exists() or args.force:
        success = download_file(zip_url, zip_path)
        if not success:
            sys.exit(1)

    # Verify integrity before extracting
    if not verify_sha256(zip_path, EXPECTED_SHA256):
        sys.exit(1)

    # Extract the zip file
    success = extract_zip(zip_path, datasets_dir)
    if not success:
        sys.exit(1)
    
    # List extracted contents
    if problem_instances_dir.exists():
        print(f"\nExtracted contents in {problem_instances_dir}:")
        for item in sorted(problem_instances_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name} ({size_mb:.1f} MB)")
            else:
                print(f"  {item.name}/ (directory)")
    
    # Cleanup if requested
    if args.cleanup and zip_path.exists():
        print(f"\nCleaning up: removing {zip_path}")
        zip_path.unlink()
        print("Cleanup completed.")
    
    print(f"\nData is ready in {datasets_dir}")
    print("\nAvailable problem files:")
    if problem_instances_dir.exists():
        problem_files = sorted([f for f in problem_instances_dir.iterdir() if f.suffix == '.pkl'])
        for problem_file in problem_files:
            print(f"  {problem_file.name}")


if __name__ == "__main__":
    main()