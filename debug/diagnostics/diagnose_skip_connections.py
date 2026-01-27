#!/usr/bin/env python3
"""
Diagnostic script to identify issues with skip connections implementation
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lodestar_with_skip_connections import LodeSTARWithSkipConnections
import deeptrack.deeplay as dl


def diagnose_skip_connections():
    """Diagnose potential issues with skip connections"""
    
    print("=" * 60)
    print("Skip Connections Diagnostic")
    print("=" * 60)
    
    # Create model
    model = LodeSTARWithSkipConnections(
        n_transforms=4,
        optimizer=dl.Adam(lr=0.0001)
    ).build()
    
    print("✅ Model created successfully")
    
    # Test with different input sizes
    test_inputs = [
        torch.randn(1, 1, 64, 64),
        torch.randn(2, 1, 64, 64),  # Batch size 2
        torch.randn(1, 1, 128, 128)
    ]
    
    for i, input_tensor in enumerate(test_inputs):
        print(f"\n--- Test {i+1}: Input shape {input_tensor.shape} ---")
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"✅ Forward pass successful")
            print(f"   Input: {input_tensor.shape}")
            print(f"   Output: {output.shape}")
            
            # Check for NaN or Inf values
            if torch.isnan(output).any():
                print("❌ NaN values detected in output!")
                return False
            
            if torch.isinf(output).any():
                print("❌ Inf values detected in output!")
                return False
            
            # Check output range
            print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check if output is all zeros (potential issue)
            if torch.allclose(output, torch.zeros_like(output), atol=1e-6):
                print("❌ Output is all zeros - this could cause loss issues!")
                return False
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    # Test gradient flow
    print(f"\n--- Testing Gradient Flow ---")
    try:
        input_tensor = torch.randn(1, 1, 64, 64, requires_grad=True)
        output = model(input_tensor)
        
        # Compute a simple loss
        loss = output.sum()
        loss.backward()
        
        print(f"✅ Gradient computation successful")
        print(f"   Loss: {loss.item():.4f}")
        
        # Check if gradients are flowing
        grad_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        
        grad_norm = grad_norm ** 0.5
        print(f"   Gradient norm: {grad_norm:.4f}")
        print(f"   Parameters with gradients: {param_count}")
        
        if grad_norm < 1e-6:
            print("❌ Very small gradient norm - potential gradient vanishing!")
            return False
            
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False
    
    # Test skip connection activations
    print(f"\n--- Testing Skip Connection Activations ---")
    try:
        input_tensor = torch.randn(1, 1, 64, 64)
        
        # Hook to capture intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks for skip connection layers
        hooks = []
        if hasattr(model, 'skip_conv_0_to_4'):
            hooks.append(model.skip_conv_0_to_4.register_forward_hook(hook_fn('skip_conv_0_to_4')))
        if hasattr(model, 'skip_conv_1_to_5'):
            hooks.append(model.skip_conv_1_to_5.register_forward_hook(hook_fn('skip_conv_1_to_5')))
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"✅ Skip connection activations captured")
        for name, activation in activations.items():
            print(f"   {name}: {activation.shape}, range: [{activation.min().item():.4f}, {activation.max().item():.4f}]")
            
    except Exception as e:
        print(f"❌ Skip connection activation test failed: {e}")
        return False
    
    # Test LodeSTAR-specific functionality
    print(f"\n--- Testing LodeSTAR Functionality ---")
    try:
        # Test with multiple transforms (LodeSTAR specific)
        input_tensor = torch.randn(4, 1, 64, 64)  # 4 transforms
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"✅ Multi-transform forward pass successful")
        print(f"   Input: {input_tensor.shape}")
        print(f"   Output: {output.shape}")
        
        # Check if outputs are reasonable for LodeSTAR
        # LodeSTAR should output displacement fields and confidence
        dx = output[:, 0]  # x displacement
        dy = output[:, 1]  # y displacement
        rho = output[:, 2]  # confidence
        
        print(f"   Δx range: [{dx.min().item():.4f}, {dx.max().item():.4f}]")
        print(f"   Δy range: [{dy.min().item():.4f}, {dy.max().item():.4f}]")
        print(f"   ρ range: [{rho.min().item():.4f}, {rho.max().item():.4f}]")
        
        # Check if confidence values are reasonable (should be positive)
        if rho.min().item() < -1.0 or rho.max().item() > 1.0:
            print("❌ Confidence values seem unreasonable!")
            return False
            
    except Exception as e:
        print(f"❌ LodeSTAR functionality test failed: {e}")
        return False
    
    print(f"\n✅ All diagnostic tests passed!")
    return True


def compare_with_default():
    """Compare skip connections model with default model"""
    
    print(f"\n--- Comparison with Default LodeSTAR ---")
    
    try:
        # Create both models
        skip_model = LodeSTARWithSkipConnections(
            n_transforms=4,
            optimizer=dl.Adam(lr=0.0001)
        ).build()
        
        default_model = dl.LodeSTAR(
            n_transforms=4,
            optimizer=dl.Adam(lr=0.0001)
        ).build()
        
        # Initialize default model
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 64, 64)
            _ = default_model(dummy_input)
        
        # Test with same input
        input_tensor = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            skip_output = skip_model(input_tensor)
            default_output = default_model(input_tensor)
        
        print(f"✅ Both models forward pass successful")
        print(f"   Skip connections output: {skip_output.shape}")
        print(f"   Default output: {default_output.shape}")
        
        # Compare outputs
        output_diff = torch.abs(skip_output - default_output).mean()
        print(f"   Mean absolute difference: {output_diff.item():.6f}")
        
        if output_diff.item() < 1e-6:
            print("❌ Outputs are nearly identical - skip connections might not be working!")
            return False
        
        print(f"✅ Models produce different outputs as expected")
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


if __name__ == "__main__":
    print("LodeSTAR Skip Connections Diagnostic Tool")
    print("=" * 60)
    
    # Run diagnostics
    test1_passed = diagnose_skip_connections()
    test2_passed = compare_with_default()
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All diagnostics passed! Skip connections should work correctly.")
        print("If you're still seeing loss issues, the problem might be:")
        print("1. Training hyperparameters")
        print("2. Data preprocessing")
        print("3. Loss function implementation")
        print("4. Learning rate or optimization settings")
    else:
        print(f"\n❌ Diagnostics failed! Issues found with skip connections implementation.")
        print("The skip connections architecture needs to be fixed before training.")
