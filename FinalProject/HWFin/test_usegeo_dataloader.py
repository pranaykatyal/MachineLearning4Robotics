#!/usr/bin/env python3
"""
Test UseGeo DataLoader

This script tests if the UseGeo dataloader can successfully load a single batch.
"""

import tensorflow as tf
import sys
import os

# Add M4Depth to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'M4Depth'))

from dataloaders import get_loader, DataloaderParameters

def test_usegeo_dataloader():
    print("="*60)
    print("Testing UseGeo DataLoader")
    print("="*60)
    
    # Get dataloader
    print("\n1. Loading UseGeo dataloader...")
    loader = get_loader("usegeo")
    print(f"   Loaded: {loader}")
    
    # Configure settings
    print("\n2. Configuring dataloader settings...")
    settings = DataloaderParameters(
        db_path_config={
            'usegeo': os.path.expanduser('~/MachineLearning4Robotics/HWFin/')
        },
        records_path=os.path.expanduser('~/MachineLearning4Robotics/HWFin/M4Depth/records/usegeo/'),
        db_seq_len=4,      # Sequence length in database
        seq_len=4,         # Sequence length for network
        augment=False      # No augmentation for testing
    )
    print(f"   Dataset path: {settings.db_path_config['usegeo']}")
    print(f"   Records path: {settings.records_path}")
    
    # Build dataset
    print("\n3. Building dataset...")
    try:
        dataset = loader.get_dataset(
            usecase="eval",
            settings=settings,
            batch_size=1,
            out_size=[384, 384],
            crop=False
        )
        print(f"   Dataset created successfully!")
        print(f"   Dataset cardinality: {dataset.cardinality().numpy()}")
    except Exception as e:
        print(f"   ERROR building dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load one batch
    print("\n4. Loading first batch...")
    try:
        for batch in dataset.take(1):
            print(f"   Batch loaded successfully!")
            print(f"   Keys: {batch.keys()}")
            print(f"   RGB shape: {batch['RGB_im'].shape}")
            print(f"   Depth shape: {batch['depth'].shape}")
            print(f"   Rotation shape: {batch['rot'].shape}")
            print(f"   Translation shape: {batch['trans'].shape}")
            print(f"   Camera focal: {batch['camera']['f']}")
            print(f"   Camera center: {batch['camera']['c']}")
            
    except Exception as e:
        print(f"   ERROR loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("SUCCESS! DataLoader working correctly!")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_usegeo_dataloader()
    sys.exit(0 if success else 1)