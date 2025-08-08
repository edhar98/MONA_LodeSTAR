#!/usr/bin/env python3
"""
Training script for LodeSTAR model
Supports both single-particle and multi-particle training
"""

import os
import sys
import argparse
from train_enhanced import main as train_main


def run_single_particle_training(particle_type):
    """Run training on a single particle type"""
    
    print(f"Starting single-particle training for {particle_type}")
    
    # Update config for single particle
    import utils
    config = utils.load_yaml('config.yaml')
    config['sample'] = particle_type
    config['training_mode'] = 'single_particle'
    
    # Save updated config
    try:
        utils.save_yaml(config, 'config.yaml')
    except Exception as e:
        print(f"Warning: Could not save updated config: {e}")
    
    # Run training
    train_main()


def run_multi_particle_training():
    """Run training on all particle types"""
    
    print("Starting multi-particle training for all particle types")
    
    # Update config for multi-particle
    import utils
    config = utils.load_yaml('config.yaml')
    config['training_mode'] = 'multi_particle'
    
    # Save updated config
    try:
        utils.save_yaml(config, 'config.yaml')
    except Exception as e:
        print(f"Warning: Could not save updated config: {e}")
    
    # Run training
    train_main()


def main():
    parser = argparse.ArgumentParser(description='Train LodeSTAR model')
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi',
                       help='Training mode: single particle or multi-particle')
    parser.add_argument('--particle', choices=['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod'],
                       default='Janus', help='Particle type for single-particle training')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_particle_training(args.particle)
    else:
        run_multi_particle_training()


if __name__ == '__main__':
    main() 