#!/usr/bin/env python3
"""
Test script to verify that training works without validation dataloader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
from train_single_particle import train_single_particle_model

def test_no_validation_training():
    """Test training functionality without validation dataloader"""
    
    print("Testing Training Setup Without Validation")
    print("=" * 50)
    
    # Load config
    with open('src/config_debug.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for quick test
    config['max_epochs'] = 1
    config['length'] = 10
    config['batch_size'] = 2
    config['devices'] = 1
    config['strategy'] = 'auto'
    
    print(f"Config: {config['lodestar_version']}")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Training samples: {config['length']}")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    # Test with Janus particle
    particle_type = "Janus"
    print(f"Testing with particle type: {particle_type}")
    
    try:
        # Test training functionality
        model_path, checkpoint_path = train_single_particle_model(particle_type, config)
        
        print("✅ SUCCESS: Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_no_validation_training()
    if success:
        print("\n🎉 All tests passed! Training setup is working correctly.")
    else:
        print("\n💥 Tests failed. Check the error messages above.")
        sys.exit(1)
