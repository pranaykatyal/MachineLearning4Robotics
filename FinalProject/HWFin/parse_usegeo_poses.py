#!/usr/bin/env python3
"""
Parse UseGeo Camera Poses
Converts UseGeo camera data to M4Depth CSV format

Input:
  - eors_couple.txt: Camera poses (X0, Y0, Z0, omega, phi, kappa) + GPSTime
  - Image_orientations_dataset1.xyz: Camera intrinsics (c, x0, y0)

Output:
  - poses.csv: timestamp,qw,qx,qy,qz,tx,ty,tz
  - intrinsics.txt: fx fy cx cy
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import argparse
from pathlib import Path


def euler_to_quaternion(omega, phi, kappa):
    """
    Convert Euler angles (omega, phi, kappa) in DEGREES to quaternion (qw, qx, qy, qz)
    
    UseGeo uses:
    - omega: rotation around X-axis (roll)
    - phi: rotation around Y-axis (pitch)  
    - kappa: rotation around Z-axis (yaw)
    
    Returns: [qw, qx, qy, qz] (scalar-first convention for M4Depth)
    """
    # Convert degrees to radians
    omega_rad = np.deg2rad(omega)
    phi_rad = np.deg2rad(phi)
    kappa_rad = np.deg2rad(kappa)
    
    # Create rotation object from Euler angles (XYZ convention)
    # Note: scipy uses 'xyz' for intrinsic rotations
    r = Rotation.from_euler('xyz', [omega_rad, phi_rad, kappa_rad])
    
    # Get quaternion in scalar-first format [qw, qx, qy, qz]
    quat = r.as_quat()  # Returns [qx, qy, qz, qw]
    
    # Reorder to [qw, qx, qy, qz] (scalar-first)
    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    
    return qw, qx, qy, qz


def parse_eors_couple(eors_file):
    """
    Parse eors_couple.txt file
    
    Format:
    #PointLabel    GPSTime    X0    Y0    Z0    omega[deg]    phi[deg]    kappa[deg]
    """
    print(f"📖 Reading poses from: {eors_file}")
    
    # Read the file, skip comment lines
    with open(eors_file, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
            
        filename = parts[0]
        gps_time = float(parts[1])
        x0 = float(parts[2])
        y0 = float(parts[3])
        z0 = float(parts[4])
        omega = float(parts[5])
        phi = float(parts[6])
        kappa = float(parts[7])
        
        # Convert Euler angles to quaternion
        qw, qx, qy, qz = euler_to_quaternion(omega, phi, kappa)
        
        data.append({
            'filename': filename,
            'timestamp': gps_time,
            'qw': qw,
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'tx': x0,
            'ty': y0,
            'tz': z0
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} camera poses")
    return df


def parse_intrinsics(intrinsics_file):
    """
    Parse Image_orientations_dataset1.xyz for camera intrinsics
    
    Format (header line):
    #label  X0  Y0  Z0  omega[deg]  phi[deg]  kappa[deg]  c  x0  y0  a3  a4  a5  a6  rho0
    
    We extract: c (focal length), x0 (principal point x), y0 (principal point y)
    """
    print(f"📖 Reading intrinsics from: {intrinsics_file}")
    
    # Read first data line (all images have same intrinsics)
    with open(intrinsics_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                c = float(parts[7])   # Focal length
                x0 = float(parts[8])  # Principal point x
                y0 = float(parts[9])  # Principal point y
                break
    
    # For M4Depth: fx, fy, cx, cy
    # UseGeo assumes square pixels, so fx = fy = c
    fx = fy = c
    cx = x0
    cy = abs(y0)  # Take absolute value (coordinate system difference)
    
    print(f"✅ Camera intrinsics:")
    print(f"   fx = {fx:.3f}")
    print(f"   fy = {fy:.3f}")
    print(f"   cx = {cx:.3f}")
    print(f"   cy = {cy:.3f}")
    
    return fx, fy, cx, cy


def save_poses_csv(df, output_file):
    """Save poses to M4Depth CSV format"""
    print(f"💾 Saving poses to: {output_file}")
    
    # Select columns in M4Depth order
    output_df = df[['timestamp', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(output_df)} poses")


def save_intrinsics_txt(fx, fy, cx, cy, output_file):
    """Save intrinsics to simple text file"""
    print(f"💾 Saving intrinsics to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    
    print(f"✅ Saved intrinsics")


def main():
    parser = argparse.ArgumentParser(description='Parse UseGeo camera poses to M4Depth format')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset folder (e.g., Dataset_1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as dataset)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎯 UseGeo Pose Parser")
    print(f"{'='*60}\n")
    
    # Input files
    eors_file = dataset_path / 'Camera_Inputs' / 'eors_couple.txt'
    
    # Auto-detect intrinsics file (Image_orientations_dataset1.xyz, dataset2.xyz, etc.)
    intrinsics_files = list(dataset_path.glob('Image_orientations_dataset*.xyz'))
    if not intrinsics_files:
        print(f"Error: No Image_orientations_dataset*.xyz file found in {dataset_path}")
        return
    intrinsics_file = intrinsics_files[0]
    
    # Check files exist
    if not eors_file.exists():
        print(f"❌ Error: {eors_file} not found!")
        return
    if not intrinsics_file.exists():
        print(f"❌ Error: {intrinsics_file} not found!")
        return
    
    # Parse camera poses
    poses_df = parse_eors_couple(eors_file)
    
    # Parse intrinsics
    fx, fy, cx, cy = parse_intrinsics(intrinsics_file)
    
    # Save outputs
    poses_csv = output_dir / 'poses.csv'
    intrinsics_txt = output_dir / 'intrinsics.txt'
    
    save_poses_csv(poses_df, poses_csv)
    save_intrinsics_txt(fx, fy, cx, cy, intrinsics_txt)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  📄 {poses_csv}")
    print(f"  📄 {intrinsics_txt}")
    print(f"\nNext steps:")
    print(f"  1. Verify poses.csv format")
    print(f"  2. Create UseGeo dataloader")
    print(f"  3. Test loading 1 batch")
    print()


if __name__ == '__main__':
    main()