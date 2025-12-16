#!/usr/bin/env python3
"""
Generate CSV records for UseGeo datasets

Creates tab-separated CSV files compatible with M4Depth dataloader format.

CSV Format (tab-separated):
id	timestamp	qw	qx	qy	qz	tx	ty	tz	rgb_image	depth_map

Where:
- id: Frame index within trajectory (0 = first frame, marks new trajectory)
- timestamp: GPS time
- qw, qx, qy, qz: Rotation quaternion
- tx, ty, tz: Translation vector
- rgb_image: Relative path to RGB image
- depth_map: Relative path to depth TIFF
"""

import pandas as pd
import argparse
from pathlib import Path
import os


def create_csv_records(dataset_path, output_dir):
    """Generate CSV records for a single dataset
    
    Args:
        dataset_path: Path to Dataset_N folder
        output_dir: Where to save CSV file
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load poses
    poses_file = dataset_path / 'poses.csv'
    if not poses_file.exists():
        print(f"ERROR: {poses_file} not found!")
        return
    
    poses_df = pd.read_csv(poses_file)
    print(f"Loaded {len(poses_df)} poses from {poses_file}")
    
    # Build records
    records = []
    
    for idx, row in poses_df.iterrows():
        # Extract filename from first pose (assumes format: YYYY-MM-DD_HH-MM-SS_*.jpg)
        # We need to match this with actual image files
        
        # For now, construct expected paths
        # RGB: Undistorted_images_full_res/*.jpg
        # Depth: Depth_resized/depth_maps/*.tiff
        
        # Get image list (we'll match by index since poses.csv is ordered)
        rgb_dir = dataset_path / 'Undistorted_images_full_res'
        depth_dir = dataset_path / 'Depth_resized' / 'depth_maps'
        
        if idx == 0:
            # Get sorted file lists
            rgb_files = sorted(rgb_dir.glob('*.jpg'))
            depth_files = sorted(depth_dir.glob('*.tiff'))
            
            print(f"Found {len(rgb_files)} RGB images")
            print(f"Found {len(depth_files)} depth maps")
            
            if len(rgb_files) != len(poses_df):
                print(f"WARNING: Mismatch! {len(rgb_files)} images vs {len(poses_df)} poses")
            if len(depth_files) != len(poses_df):
                print(f"WARNING: Mismatch! {len(depth_files)} depths vs {len(poses_df)} poses")
        
        # Get relative paths (including Dataset_N prefix!)
        if idx < len(rgb_files):
            # Get path relative to the parent of dataset_path (HWFin/)
            rgb_rel = str(rgb_files[idx].relative_to(dataset_path.parent))
        else:
            print(f"ERROR: No RGB file for pose {idx}")
            continue
        
        if idx < len(depth_files):
            depth_rel = str(depth_files[idx].relative_to(dataset_path.parent))
        else:
            print(f"WARNING: No depth file for pose {idx}")
            depth_rel = ""
        
        record = {
            'id': idx,
            'timestamp': row['timestamp'],
            'qw': row['qw'],
            'qx': row['qx'],
            'qy': row['qy'],
            'qz': row['qz'],
            'tx': row['tx'],
            'ty': row['ty'],
            'tz': row['tz'],
            'rgb_image': rgb_rel,
            'depth_map': depth_rel
        }
        
        records.append(record)
    
    # Create DataFrame
    records_df = pd.DataFrame(records)
    
    # Save to CSV (tab-separated)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / f'{dataset_name}.csv'
    records_df.to_csv(csv_file, sep='\t', index=False)
    
    print(f"\nSaved {len(records_df)} records to {csv_file}")
    print(f"\nFirst 3 records:")
    print(records_df.head(3))
    print(f"\nLast 3 records:")
    print(records_df.tail(3))


def main():
    parser = argparse.ArgumentParser(description='Generate CSV records for UseGeo datasets')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Paths to Dataset_1, Dataset_2, Dataset_3')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("UseGeo CSV Records Generator")
    print("="*60)
    
    for dataset_path in args.datasets:
        create_csv_records(dataset_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print("SUCCESS! All CSV records created")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Copy CSV files to M4Depth/records/usegeo/")
    print(f"2. Update M4Depth config with UseGeo paths")
    print(f"3. Test dataloader loading")
    print()


if __name__ == '__main__':
    main()