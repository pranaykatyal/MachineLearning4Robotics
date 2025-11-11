#!/usr/bin/env python3
"""
Diagnostic script to check what dubinEHF3d is returning
"""
import numpy as np
from dubinEHF3d import dubinEHF3d

print("=" * 60)
print("DUBINS PATH DIAGNOSTIC")
print("=" * 60)

# Test a simple case
turn_radius = 60
path_step_size = 10
start_heading = 0  # degrees -> radians
climb_angle = 15 * np.pi / 180
goal_x = 300
goal_y = 200

print(f"\nTest Case:")
print(f"  Start: (0, 0, 0)")
print(f"  Start Heading: 0°")
print(f"  Goal: ({goal_x}, {goal_y})")
print(f"  Climb Angle: 15°")
print(f"  Turn Radius: {turn_radius}")
print(f"  Step Size: {path_step_size}")

path, end_heading, num_points = dubinEHF3d(
    0, 0, 0,
    start_heading,
    goal_x, goal_y,
    turn_radius, path_step_size,
    climb_angle, 
    0  # data_count
)

print(f"\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)

if path is None:
    print("❌ ERROR: Path is None!")
else:
    print(f"✓ Path generated successfully")
    print(f"  Path shape: {path.shape}")
    print(f"  Number of points: {num_points}")
    print(f"  End heading: {end_heading}")
    
    print(f"\n  First 5 points:")
    for i in range(min(5, len(path))):
        print(f"    [{i}]: ({path[i][0]:.2f}, {path[i][1]:.2f}, {path[i][2]:.2f})")
    
    print(f"\n  Last 5 points:")
    for i in range(max(0, len(path) - 5), len(path)):
        print(f"    [{i}]: ({path[i][0]:.2f}, {path[i][1]:.2f}, {path[i][2]:.2f})")
    
    print(f"\n  Start point: ({path[0][0]:.2f}, {path[0][1]:.2f}, {path[0][2]:.2f})")
    print(f"  End point:   ({path[-1][0]:.2f}, {path[-1][1]:.2f}, {path[-1][2]:.2f})")
    
    # Check if path is moving
    if len(path) > 1:
        total_distance = 0
        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i] - path[i-1])
            total_distance += dist
        print(f"\n  Total path length: {total_distance:.2f}")
        
        dist_to_goal = np.linalg.norm(path[-1][:2] - np.array([goal_x, goal_y]))
        print(f"  Distance from endpoint to goal: {dist_to_goal:.2f}")
        
        # Check if all points are the same
        all_same = True
        for i in range(1, len(path)):
            if not np.allclose(path[i], path[0]):
                all_same = False
                break
        
        if all_same:
            print("\n  ❌ WARNING: All points in path are identical!")
            print("  This means the path generation is broken.")
        else:
            print("\n  ✓ Path has movement (points are different)")

print("\n" + "=" * 60)
print("CHECKING DATASET GENERATION")
print("=" * 60)

# Simulate what happens in generate_dataset
print("\nSimulating dataset generation...")
dataset_sample = []

for test_idx in range(3):
    test_cases = [
        (0, 300, 200, 15),
        (45, -250, 300, -20),
        (90, 400, -100, 25),
    ]
    
    start_heading_deg, gx, gy, climb_deg = test_cases[test_idx]
    start_heading = start_heading_deg * np.pi / 180
    climb_angle = climb_deg * np.pi / 180
    
    path, end_heading, num_points = dubinEHF3d(
        0, 0, 0,
        start_heading,
        gx, gy,
        turn_radius, path_step_size,
        climb_angle,
        test_idx
    )
    
    if path is not None and len(path) >= 5:
        actual_goal = path[-1]
        dataset_sample.append((path, actual_goal, end_heading, num_points))
        print(f"\nTest {test_idx+1}:")
        print(f"  Requested goal: ({gx}, {gy})")
        print(f"  Actual endpoint: ({actual_goal[0]:.2f}, {actual_goal[1]:.2f}, {actual_goal[2]:.2f})")
        print(f"  Path length: {num_points} points")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

if len(dataset_sample) > 0:
    # Check what would be used as training data
    sample_path, sample_goal, _, _ = dataset_sample[0]
    print(f"\nWhat gets stored in dataset:")
    print(f"  path[-1] (actual_goal): ({sample_goal[0]:.2f}, {sample_goal[1]:.2f}, {sample_goal[2]:.2f})")
    
    if np.allclose(sample_goal, [0, 0, 0]):
        print("\n❌ CRITICAL ERROR: The endpoint is [0, 0, 0]!")
        print("   This means dubinEHF3d is NOT generating proper paths.")
        print("\n   Possible causes:")
        print("   1. dubinEHF3d function has a bug")
        print("   2. Path array is initialized to zeros and never updated")
        print("   3. The function returns early without computing the path")
        print("\n   ACTION: Check the dubinEHF3d implementation!")
    else:
        print("\n✓ Endpoint looks valid (non-zero)")
else:
    print("\n❌ ERROR: No valid paths generated!")

print("\n" + "=" * 60)