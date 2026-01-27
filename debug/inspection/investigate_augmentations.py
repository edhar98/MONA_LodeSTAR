#!/usr/bin/env python3
"""
Single script to investigate augmentation pipeline issues
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import deeptrack as dt
import torch
import utils

def investigate_augmentations(config, particle_type):
    """Investigate why augmentations are not working"""
    
    print(f"\n🔍 Investigating augmentation pipeline for {particle_type}")
    
    # Load original sample
    sample_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.jpg')
    original_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    # Convert RGB to grayscale if needed
    if len(original_image.shape) == 3 and original_image.shape[-1] == 3:
        original_image = np.dot(original_image[..., :3], [0.299, 0.587, 0.114])
    
    # Add channel dimension if needed
    if len(original_image.shape) == 2:
        original_image = original_image[..., np.newaxis]
    
    print(f"📏 Original image shape: {original_image.shape}")
    print(f"📊 Original stats - Min: {original_image.min():.2f}, Max: {original_image.max():.2f}, Mean: {original_image.mean():.2f}")
    
    # SUSPICION 1: Check if the augmentation function is being called
    print(f"\n🧪 SUSPICION 1: Is the augmentation function being called?")
    
    # Test the current augmentation function from train_single_particle.py
    def test_augment_image(image):
        print(f"  🔄 Augmentation function called with image shape: {image.shape}")
        from scipy.ndimage import rotate
        
        # Generate random parameters
        scale = np.random.uniform(0.6, 1.8)
        rotation_degrees = np.random.uniform(0, 360)
        translation = np.random.uniform(-8, 8, 2)
        mul_factor = np.random.uniform(0.4, 2.5)
        add_value = np.random.uniform(-0.2, 0.2)
        
        print(f"  📊 Generated params - scale: {scale:.2f}, rotation: {rotation_degrees:.1f}°, translation: {translation}")
        
        # Apply augmentations step by step
        pooled = dt.AveragePooling(ksize=(1, 1, 3))(image)
        print(f"  📊 After pooling - Mean: {pooled.mean():.2f}")
        
        rotated = rotate(pooled.squeeze(), rotation_degrees, order=1, mode='constant', reshape=False)
        rotated = rotated[..., np.newaxis]
        print(f"  📊 After rotation - Mean: {rotated.mean():.2f}")
        
        scaled = dt.Affine(scale=scale, translation=translation, mode='constant')(rotated)
        print(f"  📊 After scaling - Mean: {scaled.mean():.2f}")
        
        multiplied = dt.Multiply(mul_factor)(scaled)
        added = dt.Add(add_value)(multiplied)
        print(f"  📊 After brightness - Mean: {added.mean():.2f}")
        
        noise_std = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, added.shape)
        noisy = dt.Add(noise)(added)
        print(f"  📊 After noise - Mean: {noisy.mean():.2f}")
        
        extra_scale = np.random.uniform(0.7, 1.3)
        scaled_final = dt.Multiply(extra_scale)(noisy)
        print(f"  📊 After extra scaling - Mean: {scaled_final.mean():.2f}")
        
        moved = dt.MoveAxis(-1, 0)(scaled_final)
        tensor = dt.pytorch.ToTensor(dtype=torch.float32)(moved)
        print(f"  📊 Final tensor - Mean: {tensor.mean():.2f}")
        
        return tensor
    
    # Test the augmentation function directly
    print(f"\n🧪 Testing augmentation function directly:")
    augmented1 = test_augment_image(original_image)
    print(f"\n🧪 Testing augmentation function again:")
    augmented2 = test_augment_image(original_image)
    
    # SUSPICION 2: Test the explicit pipeline approach
    print(f"\n🧪 SUSPICION 2: Testing explicit pipeline approach")
    
    # Test the explicit pipeline exactly as in train_single_particle.py
    explicit_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=lambda: np.random.uniform(0.6, 1.8),
            rotate=lambda: 2*np.pi*np.random.uniform(0, 1.0),
            translate=lambda: np.random.uniform(-8, 8, 2),
            mode='constant'
        )
        >> dt.Multiply(lambda: np.random.uniform(0.4, 2.5))
        >> dt.Add(lambda: np.random.uniform(-0.2, 0.2))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    print(f"\n🧪 Testing explicit pipeline:")
    try:
        result1 = explicit_pipeline()
        print(f"  ✅ Explicit pipeline worked - Mean: {result1.mean():.2f}")
    except Exception as e:
        print(f"  ❌ Explicit pipeline failed: {e}")
        result1 = original_image
    
    print(f"\n🧪 Testing explicit pipeline again:")
    try:
        # Test pipeline.update() approach
        explicit_pipeline.update()
        result2 = explicit_pipeline()
        print(f"  ✅ Explicit pipeline worked - Mean: {result2.mean():.2f}")
    except Exception as e:
        print(f"  ❌ Explicit pipeline failed: {e}")
        result2 = original_image
    
    # SUSPICION 3: Check if the issue is with random seed
    print(f"\n🧪 SUSPICION 3: Is the issue with random seed?")
    
    # Check if results are identical
    direct_same = np.allclose(augmented1, augmented2, atol=1e-6)
    explicit_same = np.allclose(result1, result2, atol=1e-6)
    
    print(f"  Direct calls identical: {direct_same}")
    print(f"  Explicit pipeline calls identical: {explicit_same}")
    
    # SUSPICION 4: Test dt.Value approach
    print(f"\n🧪 SUSPICION 4: Testing dt.Value approach")
    
    # Test with dt.Value for random parameters
    dt_value_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=dt.Value(lambda: np.random.uniform(0.6, 1.8)),
            rotate=dt.Value(lambda: 2*np.pi*np.random.uniform(0, 1.0)),
            translate=dt.Value(lambda: np.random.uniform(-8, 8, 2)),
            mode='constant'
        )
        >> dt.Multiply(dt.Value(lambda: np.random.uniform(0.4, 2.5)))
        >> dt.Add(dt.Value(lambda: np.random.uniform(-0.2, 0.2)))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    print(f"\n🧪 Testing dt.Value pipeline:")
    try:
        dt_value_result1 = dt_value_pipeline()
        print(f"  ✅ dt.Value pipeline worked - Mean: {dt_value_result1.mean():.2f}")
    except Exception as e:
        print(f"  ❌ dt.Value pipeline failed: {e}")
        dt_value_result1 = original_image
    
    print(f"\n🧪 Testing dt.Value pipeline again:")
    try:
        # Test pipeline.update() approach
        dt_value_pipeline.update()
        dt_value_result2 = dt_value_pipeline()
        print(f"  ✅ dt.Value pipeline worked - Mean: {dt_value_result2.mean():.2f}")
    except Exception as e:
        print(f"  ❌ dt.Value pipeline failed: {e}")
        dt_value_result2 = original_image
    
    dt_value_same = np.allclose(dt_value_result1, dt_value_result2, atol=1e-6)
    print(f"  dt.Value pipeline calls identical: {dt_value_same}")
    
    # SUSPICION 5: Test rotation specifically in pipeline
    print(f"\n🧪 SUSPICION 5: Testing rotation in pipeline vs direct calls")
    
    # Test rotation-only pipeline with original dt.Affine
    rotation_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,  # No scaling
            rotate=lambda: 2*np.pi*np.random.uniform(0, 1.0),  # Only rotation
            translate=[0, 0],  # No translation
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    # Test with different parameter names using original dt.Affine
    print(f"  Testing parameter name variations with original dt.Affine:")
    
    # Test 1: Using 'rotate' instead of 'rotation'
    rotation_pipeline_rotate = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,
            rotate=lambda: 2*np.pi*np.random.uniform(0, 1.0),  # Using 'rotate'
            translate=[0, 0],
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    # Test 2: Using direct value instead of lambda
    rotation_pipeline_direct = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,
            rotate=np.pi/4,  # Direct value (45 degrees)
            translate=[0, 0],
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    print(f"  Testing rotation-only pipeline (with 'rotate' parameter):")
    rotation_results = []
    for i in range(2):
        try:
            rotation_pipeline.update()
            result = rotation_pipeline()
            rotation_results.append(result)
            print(f"    Rotation {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Rotation {i+1}: Failed - {e}")
            rotation_results.append(original_image)
    
    print(f"  Testing rotation-only pipeline (with 'rotate' parameter - duplicate test):")
    rotation_results_rotate = []
    for i in range(2):
        try:
            rotation_pipeline_rotate.update()
            result = rotation_pipeline_rotate()
            rotation_results_rotate.append(result)
            print(f"    Rotate {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Rotate {i+1}: Failed - {e}")
            rotation_results_rotate.append(original_image)
    
    print(f"  Testing rotation-only pipeline (with direct value):")
    rotation_results_direct = []
    for i in range(2):
        try:
            rotation_pipeline_direct.update()
            result = rotation_pipeline_direct()
            rotation_results_direct.append(result)
            print(f"    Direct {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Direct {i+1}: Failed - {e}")
            rotation_results_direct.append(original_image)
    
    # Additional confirmation test: Test with 'rotate' parameter and direct value
    print(f"  Testing rotation-only pipeline (with 'rotate' parameter and direct value):")
    rotation_pipeline_rotate_direct = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,
            rotate=np.pi/4,  # Direct value (45 degrees) with 'rotate' parameter
            translate=[0, 0],
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    rotation_results_rotate_direct = []
    for i in range(2):
        try:
            rotation_pipeline_rotate_direct.update()
            result = rotation_pipeline_rotate_direct()
            rotation_results_rotate_direct.append(result)
            print(f"    Rotate Direct {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Rotate Direct {i+1}: Failed - {e}")
            rotation_results_rotate_direct.append(original_image)
    
    # Check if results are different (indicating rotation is working)
    rotation_same = len(rotation_results) >= 2 and np.allclose(rotation_results[0], rotation_results[1], atol=1e-6)
    rotate_same = len(rotation_results_rotate) >= 2 and np.allclose(rotation_results_rotate[0], rotation_results_rotate[1], atol=1e-6)
    direct_same = len(rotation_results_direct) >= 2 and np.allclose(rotation_results_direct[0], rotation_results_direct[1], atol=1e-6)
    rotate_direct_same = len(rotation_results_rotate_direct) >= 2 and np.allclose(rotation_results_rotate_direct[0], rotation_results_rotate_direct[1], atol=1e-6)
    
    print(f"  Results comparison:")
    print(f"    'rotation' parameter identical: {rotation_same}")
    print(f"    'rotate' parameter identical: {rotate_same}")
    print(f"    'rotation' direct value identical: {direct_same}")
    print(f"    'rotate' direct value identical: {rotate_direct_same}")
    
    # Check if rotation is actually happening by comparing with original
    rotation_working = len(rotation_results) >= 1 and not np.allclose(rotation_results[0], original_image, atol=1e-6)
    rotate_working = len(rotation_results_rotate) >= 1 and not np.allclose(rotation_results_rotate[0], original_image, atol=1e-6)
    direct_working = len(rotation_results_direct) >= 1 and not np.allclose(rotation_results_direct[0], original_image, atol=1e-6)
    rotate_direct_working = len(rotation_results_rotate_direct) >= 1 and not np.allclose(rotation_results_rotate_direct[0], original_image, atol=1e-6)
    
    print(f"  Rotation effectiveness:")
    print(f"    'rotation' parameter working: {rotation_working}")
    print(f"    'rotate' parameter working: {rotate_working}")
    print(f"    'rotation' direct value working: {direct_working}")
    print(f"    'rotate' direct value working: {rotate_direct_working}")
    
    # Test if 'rotation' parameter is even recognized
    print(f"\n🧪 SUSPICION 6: Testing if 'rotation' parameter is recognized")
    
    # Test with invalid parameter to see if it's ignored
    invalid_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,
            invalid_param=lambda: 999,  # Invalid parameter
            translate=[0, 0],
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    try:
        invalid_result = invalid_pipeline()
        print(f"  Invalid parameter test: Mean={invalid_result.mean():.2f}")
        print(f"  Invalid parameter ignored: {np.allclose(invalid_result, original_image, atol=1e-6)}")
    except Exception as e:
        print(f"  Invalid parameter test failed: {e}")
    
    # Test with both 'rotation' and 'rotate' to see which one is used
    both_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=1.0,
            rotation=lambda: np.pi/2,  # 90 degrees
            rotate=lambda: np.pi/4,   # 45 degrees
            translate=[0, 0],
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    try:
        both_result = both_pipeline()
        print(f"  Both parameters test: Mean={both_result.mean():.2f}")
        # Compare with known rotations
        rotation_90_pipeline = (
            dt.Value(original_image)
            >> dt.AveragePooling(ksize=(1, 1, 3))
            >> dt.Affine(
                scale=1.0,
                rotate=np.pi/2,  # 90 degrees
                translate=[0, 0],
                mode='constant'
            )
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )
        rotation_45_pipeline = (
            dt.Value(original_image)
            >> dt.AveragePooling(ksize=(1, 1, 3))
            >> dt.Affine(
                scale=1.0,
                rotate=np.pi/4,  # 45 degrees
                translate=[0, 0],
                mode='constant'
            )
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )
        
        result_90 = rotation_90_pipeline()
        result_45 = rotation_45_pipeline()
        
        matches_90 = np.allclose(both_result, result_90, atol=1e-6)
        matches_45 = np.allclose(both_result, result_45, atol=1e-6)
        
        print(f"  Matches 90° rotation: {matches_90}")
        print(f"  Matches 45° rotation: {matches_45}")
        print(f"  'rotate' parameter takes precedence: {matches_45}")
        
    except Exception as e:
        print(f"  Both parameters test failed: {e}")
    
    # Test scaling-only pipeline
    scaling_pipeline = (
        dt.Value(original_image)
        >> dt.AveragePooling(ksize=(1, 1, 3))
        >> dt.Affine(
            scale=lambda: np.random.uniform(0.6, 1.8),  # Only scaling
            rotate=0,  # No rotation
            translate=[0, 0],  # No translation
            mode='constant'
        )
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    print(f"  Testing scaling-only pipeline:")
    scaling_results = []
    for i in range(3):
        try:
            scaling_pipeline.update()
            result = scaling_pipeline()
            scaling_results.append(result)
            print(f"    Scaling {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Scaling {i+1}: Failed - {e}")
            scaling_results.append(original_image)
    
    # Test creating multiple pipeline instances (like dataloader would)
    pipeline_instances = []
    for i in range(3):
        pipeline = (
            dt.Value(original_image)
            >> dt.AveragePooling(ksize=(1, 1, 3))
            >> dt.Affine(
                scale=lambda: np.random.uniform(0.6, 1.8),
                rotate=lambda: 2*np.pi*np.random.uniform(0, 1.0),
                translate=lambda: np.random.uniform(-8, 8, 2),
                mode='constant'
            )
            >> dt.Multiply(lambda: np.random.uniform(0.4, 2.5))
            >> dt.Add(lambda: np.random.uniform(-0.2, 0.2))
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )
        pipeline_instances.append(pipeline)
    
    print(f"  Testing multiple pipeline instances:")
    results = []
    for i, pipeline in enumerate(pipeline_instances):
        try:
            result = pipeline()
            results.append(result)
            print(f"    Pipeline {i+1}: Mean={result.mean():.2f}")
        except Exception as e:
            print(f"    Pipeline {i+1}: Failed - {e}")
            results.append(original_image)
    
    # Check if multiple instances produce different results
    if len(results) >= 2:
        multi_instance_same = all(np.allclose(results[0], results[i], atol=1e-6) for i in range(1, len(results)))
        print(f"  Multiple instances identical: {multi_instance_same}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Augmentation Investigation', fontsize=16)
    
    images = [
        ("Original", original_image),
        ("Direct Call 1", augmented1.squeeze()),
        ("Direct Call 2", augmented2.squeeze()),
        ("Rotation Only 1", rotation_results[0].squeeze()),
        ("Rotation Only 2", rotation_results[1].squeeze()),
        ("Scaling Only 1", scaling_results[0].squeeze())
    ]
    
    for i, (name, image) in enumerate(images):
        row = i // 3
        col = i % 3
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'{name}\nMean: {image.mean():.2f}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = 'debug_outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'augmentation_investigation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved investigation to: {output_path}")
    
    plt.show()
    
    # Summary of suspicions
    print(f"\n📋 SUMMARY OF SUSPICIONS:")
    print(f"1. Augmentation function called: ✅ (we see the print statements)")
    print(f"2. Explicit pipeline working: {'✅' if not np.allclose(result1, original_image, atol=1e-6) else '❌'}")
    print(f"3. Explicit pipeline producing different results: {'✅' if not explicit_same else '❌'}")
    print(f"4. dt.Value pipeline working: {'✅' if not np.allclose(dt_value_result1, original_image, atol=1e-6) else '❌'}")
    print(f"5. dt.Value pipeline producing different results: {'✅' if not dt_value_same else '❌'}")
    print(f"6. Rotation-only pipeline working: {'✅' if len(rotation_results) >= 2 and not all(np.allclose(rotation_results[0], rotation_results[i], atol=1e-6) for i in range(1, len(rotation_results))) else '❌'}")
    print(f"7. Scaling-only pipeline working: {'✅' if len(scaling_results) >= 2 and not all(np.allclose(scaling_results[0], scaling_results[i], atol=1e-6) for i in range(1, len(scaling_results))) else '❌'}")
    print(f"8. Multiple instances producing different results: {'✅' if len(results) >= 2 and not multi_instance_same else '❌'}")
    print(f"9. Random seed issue: {'❌' if not direct_same else '✅'}")
    
    return output_path


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Investigate augmentation pipeline')
    parser.add_argument('--particle', type=str, default='Rod', help='Particle type')
    parser.add_argument('--config', type=str, default='src/config_debug.yaml', help='Config file')
    args = parser.parse_args()
    
    # Load configuration
    config = utils.load_yaml(args.config)
    
    # Check if particle type exists
    if args.particle not in config['samples']:
        print(f"❌ Particle type '{args.particle}' not found in config.")
        print(f"Available types: {config['samples']}")
        return
    
    # Run investigation
    try:
        output_path = investigate_augmentations(config, args.particle)
        print(f"\n✅ Investigation completed!")
        print(f"📁 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
