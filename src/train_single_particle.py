import os
import sys
import shutil
import argparse
from datetime import datetime
from deeplay import LodeSTAR
import deeptrack.deeplay as dl
import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import utils
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from wandb_logging import (
    WANDB_AVAILABLE, get_logger, get_run_id, set_summary, finish_run, TrainingMetricsCallback
)

from custom_lodestar import customLodeSTAR
from scipy.ndimage import gaussian_filter

# Setup logger with file output
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'train_single_particle_{timestamp}.log')
logger = utils.setup_logger('train_single_particle', log_file=log_file)


def create_circular_mask(image, radius, soft_edge=0):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    if soft_edge > 0:
        mask = 1 - np.clip((dist - radius) / soft_edge, 0, 1)
        mask = gaussian_filter(mask, sigma=soft_edge / 2)
    else:
        mask = (dist <= radius).astype(np.float32)
    if len(image.shape) == 3:
        mask = mask[..., np.newaxis]
    return mask


def create_single_particle_pipeline(config, particle_type):
    """Create training pipeline for a specific particle type"""
    
    # Try both .jpg and .png as possible sample image extensions
    possible_extensions = ['jpg', 'png']
    sample_path = None
    for ext in possible_extensions:
        candidate_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.{ext}')
        if os.path.exists(candidate_path):
            sample_path = candidate_path
            break
    if sample_path is None:
        # Default to .jpg for error message if neither exists
        sample_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.jpg')
    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    training_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    if len(training_image.shape) == 3 and training_image.shape[-1] == 3:
        training_image = np.dot(training_image[..., :3], [0.299, 0.587, 0.114])
    
    if len(training_image.shape) == 2:
        training_image = training_image[..., np.newaxis]
    
    training_pipeline = (
        dt.Value(training_image)
        #>> dt.AveragePooling(ksize=(config['downsample'], config['downsample'], 3))
        >> dt.Affine(
            scale=lambda: np.random.uniform(config['scale_min'], config['scale_max']),
            rotate=lambda: 2*np.pi*np.random.uniform(config['rotation_range'][0], config['rotation_range'][1]),
            translate=lambda: np.random.uniform(config['translation_range'][0], config['translation_range'][1], 2),
            mode='constant'
        )
        #>> dt.Gaussian(sigma=lambda:np.random.uniform(config['sigma_min'], config['sigma_max']))
        >> dt.Multiply(lambda: np.random.uniform(config['mul_min'], config['mul_max']))
        >> dt.Add(lambda: np.random.uniform(config['add_min'], config['add_max']))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return training_pipeline


def create_validation_pipeline(config, particle_type):
    """Create validation pipeline for a specific particle type"""
    
    # Try both .jpg and .png as possible sample image extensions
    possible_extensions = ['jpg', 'png']
    sample_path = None
    for ext in possible_extensions:
        candidate_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.{ext}')
        if os.path.exists(candidate_path):
            sample_path = candidate_path
            break
    if sample_path is None:
        # Default to .jpg for error message if neither exists
        sample_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.jpg')    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    validation_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    if len(validation_image.shape) == 3 and validation_image.shape[-1] == 3:
        validation_image = np.dot(validation_image[..., :3], [0.299, 0.587, 0.114])
    
    if len(validation_image.shape) == 2:
        validation_image = validation_image[..., np.newaxis]
    
    # Use validation-specific augmentation parameters
    val_mul_min = config.get('val_mul_min', config['mul_min'])
    val_mul_max = config.get('val_mul_max', config['mul_max'])
    val_add_min = config.get('val_add_min', config['add_min'])
    val_add_max = config.get('val_add_max', config['add_max'])
    val_scale_min = config.get('val_scale_min', config['scale_min'])
    val_scale_max = config.get('val_scale_max', config['scale_max'])
    val_rotation_range = config.get('val_rotation_range', config['rotation_range'])
    val_translation_range = config.get('val_translation_range', config['translation_range'])
    val_sigma_min = config.get('val_sigma_min', config['sigma_min'])
    val_sigma_max = config.get('val_sigma_max', config['sigma_max'])

    validation_pipeline = (
        dt.Value(validation_image)
        #>> dt.AveragePooling(ksize=(config['downsample'], config['downsample'], 3))
        >> dt.Affine(
            scale=lambda: np.random.uniform(val_scale_min, val_scale_max),
            rotate=lambda: 2*np.pi*np.random.uniform(val_rotation_range[0], val_rotation_range[1]),
            translate=lambda: np.random.uniform(val_translation_range[0], val_translation_range[1], 2),
            mode='constant'
        )
        #>> dt.Gaussian(sigma=lambda:np.random.uniform(val_sigma_min, val_sigma_max))
        >> dt.Multiply(lambda: np.random.uniform(val_mul_min, val_mul_max))
        >> dt.Add(lambda: np.random.uniform(val_add_min, val_add_max))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return validation_pipeline


def train_single_particle_model(particle_type, config, checkpoint_path=None):
    """Train a LodeSTAR model for a specific particle type"""
    
    logger.info(f"\n=== Training LodeSTAR for {particle_type} particles ===")
    
    # Configure WandB project name
    if 'wandb' not in config:
        config['wandb'] = {}
    config['wandb']['project'] = f"LodeSTAR_{particle_type}"
    config['wandb']['notes'] = f"Training LodeSTAR model for {particle_type} particle detection"
    
    # Set random seed
    L.seed_everything(config['seed'])
    
    # Setup logger (WandB if available, no-op otherwise)
    exp_logger = get_logger(config, particle_type)
    
    # Create checkpoint directory
    dir_export = config.get('dir_export', 'lightning_logs')
    run_id = get_run_id(exp_logger)
    dir_checkpoint = os.path.join(dir_export, run_id, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)
    
    # Create models directory for final outputs
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create data generation pipeline
    training_pipeline = create_single_particle_pipeline(config, particle_type)
    validation_pipeline = create_validation_pipeline(config, particle_type)
    
    # Create datasets
    training_dataset = dt.pytorch.Dataset(
        training_pipeline, 
        length=config['length'], 
        replace=False
    )
    
    validation_dataset = dt.pytorch.Dataset(
        validation_pipeline, 
        length=config['length'] // 4,  # Smaller validation set
        replace=False
    )
    
    logger.info(f"Created training dataset with {config['length']} samples")
    logger.info(f"Created validation dataset with {config['length'] // 4} samples")
    
    # Log augmentation parameters
    logger.info(f"Training augmentation: mul({config['mul_min']:.2f}, {config['mul_max']:.2f}), add({config['add_min']:.2f}, {config['add_max']:.2f})")
    logger.info(f"Validation augmentation: mul({config.get('val_mul_min', config['mul_min']):.2f}, {config.get('val_mul_max', config['mul_max']):.2f}), add({config.get('val_add_min', config['add_min']):.2f}, {config.get('val_add_max', config['add_max']):.2f})")
    
    # Create dataloaders
    train_dataloader = dl.DataLoader(
        training_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_dataloader = dl.DataLoader(
        validation_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    logger.info(f"Training batches: {len(train_dataloader)}")
    
    # Get sample path for logging original image
    sample_path = None
    for ext in ['jpg', 'png']:
        candidate = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.{ext}')
        if os.path.exists(candidate):
            sample_path = candidate
            break
    
    # Setup callbacks (after dataloaders for sample logging)
    source_files = [
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(__file__), 'config.yaml'),
    ]
    callbacks = [
        ModelCheckpoint(
            monitor='val_within_image_disagreement',
            mode='min',
            filename=f'{particle_type}-{{epoch}}-{{step}}-{{val_loss:.2f}}',
            save_top_k=3,
            every_n_epochs=1,
            dirpath=dir_checkpoint
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TrainingMetricsCallback(particle_type, train_dataloader, sample_path, logger, source_files),
    ]
    
    # Initialize LodeSTAR model based on configuration
    if config['lodestar_version'] == 'default':
        lodestar = dl.LodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
    elif config['lodestar_version'] == 'skip_connections':
        from lodestar_with_skip_connections import LodeSTARWithSkipConnections
        lodestar = LodeSTARWithSkipConnections(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
    else:
        lodestar = customLodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
    

    
    # Initialize model parameters for distributed training
    logger.info("Initializing model parameters with dummy forward pass...")
    try:
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.randn(1, 1, 64, 64)
            # Run forward pass to initialize parameters
            _ = lodestar(dummy_input)
        logger.info("Model parameters initialized successfully")
    except Exception as e:
        logger.warning(f"Warning: Dummy forward pass failed: {e}")
        logger.info("Continuing with training...")
    
    # Configure PyTorch Lightning trainer
    trainer = dl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=config['lightning']['accelerator'],
        devices=config['devices'],
        strategy=config.get('strategy', 'auto'),  # Add strategy parameter
        precision=config['lightning']['precision'],
        gradient_clip_val=config['lightning']['gradient_clip_val'],
        accumulate_grad_batches=config['lightning']['accumulate_grad_batches'],
        log_every_n_steps=config['lightning']['log_every_n_steps'],
        
        # Configure validation
        limit_val_batches=1.0,  # Use full validation set
        check_val_every_n_epoch=1,  # Validate every epoch
        
        logger=exp_logger if WANDB_AVAILABLE else True,
        callbacks=callbacks
    )
    
    # Save training configuration
    utils.save_yaml(config, os.path.join(dir_checkpoint, 'config.yaml'))
    
    # Log training configuration
    logger.info(f"Training on particle type: {particle_type}")
    set_summary("particle_type", particle_type)
    set_summary("batches_per_epoch", len(train_dataloader))
    
    # Start training process
    logger.info(f"Starting LodeSTAR training for {particle_type}...")
    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(lodestar, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(lodestar, train_dataloader, val_dataloader)
    

    
    # Save trained model weights
    model_path = os.path.join(dir_checkpoint, f"{particle_type}_weights.pth")
    torch.save(lodestar.state_dict(), model_path)
    
    # Save final training checkpoint
    final_checkpoint = os.path.join(dir_checkpoint, f'{particle_type}_final_epoch.ckpt')
    trainer.save_checkpoint(final_checkpoint)
    
    # Create organized model storage
    run_models_dir = os.path.join(models_dir, run_id)
    os.makedirs(run_models_dir, exist_ok=True)
    
    # Archive final model and configuration
    final_model_path = os.path.join(run_models_dir, f"{particle_type}_weights.pth")
    final_config_path = os.path.join(run_models_dir, 'config.yaml')
    
    # Archive model weights
    shutil.copy2(model_path, final_model_path)
    
    # Archive configuration
    shutil.copy2(os.path.join(dir_checkpoint, 'config.yaml'), final_config_path)
    
    logger.info(f"Training complete! Model saved to {model_path}")
    logger.info(f"Final model copied to {final_model_path}")
    logger.info(f"Config copied to {final_config_path}")
    
    # Log training completion summary
    set_summary("model_size_mb", os.path.getsize(model_path) / (1024 * 1024))
    set_summary("final_model_path", final_model_path)
    
    # Finish experiment logger
    finish_run(exp_logger)
    return final_model_path, final_checkpoint


def main():
    """Train separate models for each particle type or a specific particle type"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LodeSTAR model for particle detection')
    parser.add_argument('--particle', type=str, help='Specific particle type to train (e.g., Janus, Ring, Spot, Ellipse, Rod)')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from')
    args = parser.parse_args()
    
    # Load training configuration
    config = utils.load_yaml(args.config)
    
    # Define particle types
    if args.particle:
        # Train only the specified particle type
        if args.particle not in config['samples']:
            logger.error(f"Particle type '{args.particle}' not found in config. Available types: {config['samples']}")
            return
        particle_types = [args.particle]
        logger.info(f"Training only {args.particle} model")
    else:
        # Train all particle types
        particle_types = config['samples']
        logger.info(f"Training all particle types: {particle_types}")
    
    # Load previous training results
    summary_path = 'trained_models_summary.yaml'
    existing_models = {}
    if os.path.exists(summary_path):
        try:
            existing_models = utils.load_yaml(summary_path)
            logger.info(f"Loaded existing training summary from {summary_path}")
        except Exception as e:
            logger.warning(f"Could not load existing training summary: {e}")
            existing_models = {}
    
    # Train models for specified particle types
    trained_models = existing_models.copy()
    successful_training = {}
    
    for particle_type in particle_types:
        try:
            final_model_path, checkpoint_path = train_single_particle_model(particle_type, config, args.checkpoint)
            
            # Record successful training
            new_model_entry = {
                'model_path': final_model_path,
                'checkpoint_path': checkpoint_path,
                'models_dir': os.path.dirname(final_model_path)
            }
            
            # Record training success
            successful_training[particle_type] = new_model_entry
            logger.info(f"Successfully trained {particle_type} model")
            
        except Exception as e:
            logger.error(f"Failed to train {particle_type} model: {e}")
            continue
    
    # Update training summary for successful runs
    for particle_type, new_model_entry in successful_training.items():
        # Check if this particle type already exists
        if particle_type in trained_models:
            
            # Preserve previous model versions
            if 'additional_models' not in trained_models[particle_type]:
                trained_models[particle_type]['additional_models'] = []
            
            # Archive previous model
            existing_entry = {
                'model_path': trained_models[particle_type]['model_path'],
                'checkpoint_path': trained_models[particle_type]['checkpoint_path'],
                'models_dir': trained_models[particle_type]['models_dir']
            }
            trained_models[particle_type]['additional_models'].insert(0, existing_entry)
            
            # Update with new model entry
            trained_models[particle_type].update(new_model_entry)
            
            logger.info(f"Updated {particle_type} model - moved previous entry to additional_models")
        else:
            # New particle type, just add it
            trained_models[particle_type] = new_model_entry
            logger.info(f"Added new {particle_type} model")
    
    # Save updated training summary
    if successful_training:
        try:
            utils.save_yaml(trained_models, summary_path)
            logger.info(f"Training summary successfully saved to {summary_path}")
                            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"\nTraining summary saved to {summary_path}")
    else:
        logger.warning("No models were successfully trained. YAML file not updated.")
    
    logger.info(f"\n=== Training Complete ===")
    if successful_training:
        logger.info(f"Successfully trained {len(successful_training)} models:")
        for particle_type, paths in successful_training.items():
            logger.info(f"  - {particle_type}: {paths['model_path']}")
            logger.info(f"    Models directory: {paths['models_dir']}")
            if particle_type in trained_models and 'additional_models' in trained_models[particle_type]:
                logger.info(f"    Additional models: {len(trained_models[particle_type]['additional_models'])}")
    else:
        logger.info("No models were successfully trained.")


if __name__ == '__main__':
    main() 