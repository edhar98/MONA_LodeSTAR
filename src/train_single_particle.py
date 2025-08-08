import os
import sys
import shutil
from deeplay import LodeSTAR
import deeptrack.deeplay as dl
import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import utils
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup logger
logger = utils.setup_logger('train_single_particle')


class LodeSTARMetricsCallback(Callback):
    """Custom callback to track LodeSTAR-specific metrics"""
    
    def __init__(self, particle_type):
        super().__init__()
        self.particle_type = particle_type
        self.parameters_logged = False
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start metrics"""
        wandb.log({
            f"{self.particle_type}/epoch": trainer.current_epoch,
            f"{self.particle_type}/learning_rate": trainer.optimizers[0].param_groups[0]['lr'],
        })
        
        # Log model parameters after first epoch (when model is initialized)
        if not self.parameters_logged and trainer.current_epoch == 0:
            try:
                total_params = sum(p.numel() for p in pl_module.parameters())
                trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
                wandb.log({
                    f"{self.particle_type}/model/total_params": total_params,
                    f"{self.particle_type}/model/trainable_params": trainable_params,
                })
                logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
                self.parameters_logged = True
            except Exception as e:
                logger.warning(f"Warning: Could not count model parameters: {e}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch end metrics"""
        # Get training metrics
        train_metrics = trainer.callback_metrics
        
        # Log key metrics
        metrics_to_log = {}
        for key, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics_to_log[f"{self.particle_type}/train/{key}"] = value.item()
            else:
                metrics_to_log[f"{self.particle_type}/train/{key}"] = value
        
        wandb.log(metrics_to_log)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics"""
        # Get validation metrics
        val_metrics = trainer.callback_metrics
        
        # Log key metrics
        metrics_to_log = {}
        for key, value in val_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics_to_log[f"{self.particle_type}/val/{key}"] = value.item()
            else:
                metrics_to_log[f"{self.particle_type}/val/{key}"] = value
        
        wandb.log(metrics_to_log)
    
    def on_train_end(self, trainer, pl_module):
        """Log final training metrics"""
        wandb.log({
            f"{self.particle_type}/final_train_loss": trainer.callback_metrics.get('train_loss', 0),
            f"{self.particle_type}/final_val_loss": trainer.callback_metrics.get('val_loss', 0),
            f"{self.particle_type}/training_complete": True,
        })


def create_single_particle_pipeline(config, particle_type):
    """Create training pipeline for a specific particle type"""
    
    sample_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.jpg')
    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    training_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    # Convert RGB to grayscale if needed
    if len(training_image.shape) == 3 and training_image.shape[-1] == 3:
        # Convert RGB to grayscale using luminance formula
        training_image = np.dot(training_image[..., :3], [0.299, 0.587, 0.114])
    
    # Add channel dimension if needed
    if len(training_image.shape) == 2:
        training_image = training_image[..., np.newaxis]
    
    training_pipeline = (
        dt.Value(training_image)
        >> dt.Multiply(lambda: np.random.uniform(config['mul_min'], config['mul_max']))
        >> dt.Add(lambda: np.random.uniform(config['add_min'], config['add_max']))
        >> dt.MoveAxis(-1, 0)  # Move channel to first dimension
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return training_pipeline


def create_validation_pipeline(config, particle_type):
    """Create validation pipeline for a specific particle type"""
    
    sample_path = os.path.join(config['data_dir'], 'Samples', particle_type, f'{particle_type}.jpg')
    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    validation_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    # Convert RGB to grayscale if needed
    if len(validation_image.shape) == 3 and validation_image.shape[-1] == 3:
        # Convert RGB to grayscale using luminance formula
        validation_image = np.dot(validation_image[..., :3], [0.299, 0.587, 0.114])
    
    # Add channel dimension if needed
    if len(validation_image.shape) == 2:
        validation_image = validation_image[..., np.newaxis]
    
    # Use validation-specific augmentation parameters
    val_mul_min = config.get('val_mul_min', config['mul_min'])
    val_mul_max = config.get('val_mul_max', config['mul_max'])
    val_add_min = config.get('val_add_min', config['add_min'])
    val_add_max = config.get('val_add_max', config['add_max'])
    
    validation_pipeline = (
        dt.Value(validation_image)
        >> dt.Multiply(lambda: np.random.uniform(val_mul_min, val_mul_max))
        >> dt.Add(lambda: np.random.uniform(val_add_min, val_add_max))
        >> dt.MoveAxis(-1, 0)  # Move channel to first dimension
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return validation_pipeline


def train_single_particle_model(particle_type, config):
    """Train a LodeSTAR model for a specific particle type"""
    
    logger.info(f"\n=== Training LodeSTAR for {particle_type} particles ===")
    
    # Update wandb project name for this particle type
    config['wandb']['project'] = f"LodeSTAR_{particle_type}"
    config['wandb']['notes'] = f"Training LodeSTAR model for {particle_type} particle detection"
    
    # Set random seed
    L.seed_everything(config['seed'])
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config=config,
        dir=config.get('wandb_log_dir', 'wandb_logs')
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        log_model=True,
        dir=config.get('wandb_log_dir', 'wandb_logs')
    )
    
    # Setup checkpoint directory
    dir_export = config.get('dir_export', 'lightning_logs')
    dir_checkpoint = os.path.join(dir_export, wandb.run.id, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)
    
    # Setup models directory for final models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='train_within_image_disagreement',
            mode='min',
            filename=f'{particle_type}-{{epoch}}-{{step}}-{{val_loss:.2f}}',
            save_top_k=3,
            every_n_epochs=1,
            dirpath=dir_checkpoint
        ),
        LearningRateMonitor(logging_interval='epoch'),
        LodeSTARMetricsCallback(particle_type),
    ]
    
    # Create training and validation pipelines
    training_pipeline = create_single_particle_pipeline(config, particle_type)
    validation_pipeline = create_validation_pipeline(config, particle_type)
    
    # Create datasets
    training_dataset = dt.pytorch.Dataset(
        training_pipeline, 
        length=config['length'], 
        replace=False
    )
    
    validation_length = config.get('validation_length', config['length'] // 4)
    validation_dataset = dt.pytorch.Dataset(
        validation_pipeline, 
        length=validation_length,
        replace=False
    )
    
    logger.info(f"Created training dataset with {config['length']} samples")
    logger.info(f"Created validation dataset with {validation_length} samples")
    
    # Log augmentation parameters
    logger.info(f"Training augmentation: mul({config['mul_min']:.2f}, {config['mul_max']:.2f}), add({config['add_min']:.2f}, {config['add_max']:.2f})")
    val_mul_min = config.get('val_mul_min', config['mul_min'])
    val_mul_max = config.get('val_mul_max', config['mul_max'])
    val_add_min = config.get('val_add_min', config['add_min'])
    val_add_max = config.get('val_add_max', config['add_max'])
    logger.info(f"Validation augmentation: mul({val_mul_min:.2f}, {val_mul_max:.2f}), add({val_add_min:.2f}, {val_add_max:.2f})")
    
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
    logger.info(f"Validation batches: {len(val_dataloader)}")
    
    # Create LodeSTAR model
    lodestar = dl.LodeSTAR(
        n_transforms=config['n_transforms'], 
        optimizer=dl.Adam(lr=config['lr'])
    ).build()
    
    # Setup trainer
    trainer = dl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=config['lightning']['accelerator'],
        devices=config['devices'],
        precision=config['lightning']['precision'],
        gradient_clip_val=config['lightning']['gradient_clip_val'],
        accumulate_grad_batches=config['lightning']['accumulate_grad_batches'],
        log_every_n_steps=config['lightning']['log_every_n_steps'],
        val_check_interval=config['lightning']['val_check_interval'],
        check_val_every_n_epoch=config['lightning']['check_val_every_n_epoch'],
        logger=wandb_logger,
        callbacks=callbacks
    )
    
    # Save configuration to checkpoint directory
    utils.save_yaml(config, os.path.join(dir_checkpoint, 'config.yaml'))
    
    # Log particle type
    logger.info(f"Training on particle type: {particle_type}")
    wandb.log({"particle_type": particle_type})
    
    # Log training configuration (without parameter counting for now)
    config_metrics = {
        "config/n_transforms": config['n_transforms'],
        "config/max_epochs": config['max_epochs'],
        "config/batch_size": config['batch_size'],
        "config/lr": config['lr'],
        "config/alpha": config['alpha'],
        "config/beta": config['beta'],
        "config/cutoff": config['cutoff'],
        "config/length": config['length'],
        "config/validation_length": validation_length,
        "config/training_augmentation/mul_min": config['mul_min'],
        "config/training_augmentation/mul_max": config['mul_max'],
        "config/training_augmentation/add_min": config['add_min'],
        "config/training_augmentation/add_max": config['add_max'],
        "config/validation_augmentation/mul_min": config.get('val_mul_min', config['mul_min']),
        "config/validation_augmentation/mul_max": config.get('val_mul_max', config['mul_max']),
        "config/validation_augmentation/add_min": config.get('val_add_min', config['add_min']),
        "config/validation_augmentation/add_max": config.get('val_add_max', config['add_max']),
        "training/batches_per_epoch": len(train_dataloader),
        "training/validation_batches": len(val_dataloader),
        "model/total_params": 0,  # Will be updated after training starts
        "model/trainable_params": 0,  # Will be updated after training starts
    }
    
    wandb.log(config_metrics)
    
    # Run training
    logger.info(f"Starting LodeSTAR training for {particle_type}...")
    trainer.fit(lodestar, train_dataloader, val_dataloader)
    
    # Save model weights to checkpoint directory
    model_path = os.path.join(dir_checkpoint, f"{particle_type}_weights.pth")
    torch.save(lodestar.state_dict(), model_path)
    
    # Save final checkpoint to checkpoint directory
    final_checkpoint = os.path.join(dir_checkpoint, f'{particle_type}_final_epoch.ckpt')
    trainer.save_checkpoint(final_checkpoint)
    
    # Create models directory with run ID name
    run_models_dir = os.path.join(models_dir, wandb.run.id)
    os.makedirs(run_models_dir, exist_ok=True)
    
    # Copy final model and config to models directory
    final_model_path = os.path.join(run_models_dir, f"{particle_type}_weights.pth")
    final_config_path = os.path.join(run_models_dir, 'config.yaml')
    
    # Copy model weights
    shutil.copy2(model_path, final_model_path)
    
    # Copy config
    shutil.copy2(os.path.join(dir_checkpoint, 'config.yaml'), final_config_path)
    
    logger.info(f"Training complete! Model saved to {model_path}")
    logger.info(f"Final model copied to {final_model_path}")
    logger.info(f"Config copied to {final_config_path}")
    
    # Log final training summary
    wandb.log({
        f"{particle_type}/training_summary/model_path": model_path,
        f"{particle_type}/training_summary/checkpoint_path": final_checkpoint,
        f"{particle_type}/training_summary/final_model_path": final_model_path,
        f"{particle_type}/training_summary/final_config_path": final_config_path,
        f"{particle_type}/training_summary/total_epochs": config['max_epochs'],
        f"{particle_type}/training_summary/training_samples": config['length'],
        f"{particle_type}/training_summary/validation_samples": validation_length,
        f"{particle_type}/training_summary/model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    })
    
    wandb.finish()
    
    return final_model_path, final_checkpoint


def main():
    """Train separate models for each particle type"""
    
    # Load configuration
    config = utils.load_yaml('src/config.yaml')
    
    # Define particle types
    particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    
    # Train models for each particle type
    trained_models = {}
    
    for particle_type in particle_types:
        try:
            final_model_path, checkpoint_path = train_single_particle_model(particle_type, config)
            trained_models[particle_type] = {
                'model_path': final_model_path,  # This is now the path in models directory
                'checkpoint_path': checkpoint_path,
                'models_dir': os.path.dirname(final_model_path)  # Add the models directory path
            }
            logger.info(f"Successfully trained {particle_type} model")
        except Exception as e:
            logger.error(f"Failed to train {particle_type} model: {e}")
            continue
    
    # Save training summary
    summary_path = 'trained_models_summary.yaml'
    utils.save_yaml(trained_models, summary_path)
    logger.info(f"\nTraining summary saved to {summary_path}")
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Successfully trained {len(trained_models)} models:")
    for particle_type, paths in trained_models.items():
        logger.info(f"  - {particle_type}: {paths['model_path']}")
        logger.info(f"    Models directory: {paths['models_dir']}")


if __name__ == '__main__':
    main() 