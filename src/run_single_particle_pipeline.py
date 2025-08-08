#!/usr/bin/env python3
"""
Single Particle Training and Testing Pipeline
Trains separate models for each particle type and evaluates them
"""

import os
import sys
import argparse
import subprocess
import time
from train_single_particle import main as train_main
from test_single_particle import main as test_main
import yaml
import utils

# Setup logger
logger = utils.setup_logger('run_single_particle_pipeline')


def run_training():
    """Run training for all particle types"""
    
    logger.info("Starting single-particle training pipeline...")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        train_main()
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.1f} seconds")
        return True
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        return False


def run_testing():
    """Run testing for all trained models"""
    
    logger.info("\nStarting single-particle testing pipeline...")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        test_main()
        testing_time = time.time() - start_time
        logger.info(f"\nTesting completed in {testing_time:.1f} seconds")
        return True
    except Exception as e:
        logger.error(f"\nTesting failed: {e}")
        return False


def check_prerequisites():
    """Check if prerequisites are met"""
    
    logger.info("Checking prerequisites...")
    
    # Load config to get data_dir
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_dir']
    
    # Check if sample images exist
    particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    missing_samples = []

    for particle_type in particle_types:
        sample_path = os.path.join(data_dir, 'Samples', particle_type, f'{particle_type}.jpg')
        if not os.path.exists(sample_path):
            missing_samples.append(particle_type)
    
    if missing_samples:
        logger.warning(f"Missing sample images: {missing_samples}")
        logger.warning("Please run: python src/generate_samples.py")
        return False
    
    # Check if testing datasets exist
    testing_dir = os.path.join(data_dir, 'Testing')
    dataset_types = ['same_shape_same_size', 'same_shape_different_size', 
                    'different_shape_same_size', 'different_shape_different_size']
    
    missing_datasets = []
    for dataset_type in dataset_types:
        dataset_path = os.path.join(testing_dir, dataset_type)
        if not os.path.exists(dataset_path):
            missing_datasets.append(dataset_type)
        else:
            # Check if images and annotations directories exist
            images_dir = os.path.join(dataset_path, 'images')
            annotations_dir = os.path.join(dataset_path, 'annotations')
            
            if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
                missing_datasets.append(f"{dataset_type} (missing images/annotations)")
    
    if missing_datasets:
        logger.warning(f"Missing testing datasets: {missing_datasets}")
        logger.warning("Please run: python src/image_generator.py to generate testing datasets")
        return False
    
    logger.info("All prerequisites met")
    return True


def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Single Particle Training and Testing Pipeline')
    parser.add_argument('--train-only', action='store_true', help='Run only training')
    parser.add_argument('--test-only', action='store_true', help='Run only testing')
    parser.add_argument('--skip-checks', action='store_true', help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    logger.info("Single Particle LodeSTAR Pipeline")
    logger.info("=" * 50)
    
    # Check prerequisites unless skipped
    if not args.skip_checks:
        if not check_prerequisites():
            return
    
    # Determine what to run
    run_train = not args.test_only
    run_test = not args.train_only
    
    # Run training
    if run_train:
        if not run_training():
            logger.error("Training failed, stopping pipeline")
            return
    
    # Run testing
    if run_test:
        if not run_testing():
            logger.error("Testing failed")
            return
    
    logger.info("\nPipeline completed successfully!")
    logger.info("=" * 50)
    logger.info("Results:")
    logger.info("  - Training summary: trained_models_summary.yaml")
    logger.info("  - Test results: test_results_summary.yaml")
    logger.info("  - Visualizations: detection_results_*.png")


if __name__ == '__main__':
    main() 