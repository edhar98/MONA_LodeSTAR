import os
import sys
from deeplay import LodeSTAR
import deeptrack.deeplay as dl
import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import utils
import random
from torch.optim.lr_scheduler import CosineAnnealingLR


class MultiParticleLodeSTAR(dl.LodeSTAR):
    """Enhanced LodeSTAR model with multi-particle support"""
    
    def __init__(self, n_transforms, optimizer, particle_types=None, **kwargs):
        super().__init__(n_transforms=n_transforms, optimizer=optimizer, **kwargs)
        self.particle_types = particle_types or ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
        
    def configure_optimizers(self):
        """Configure optimizer with scheduler"""
        optimizer = self.optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_within_image_disagreement"
            }
        }


def create_single_particle_pipeline(config):
    """Create training pipeline for single particle type"""
    
    sample = config['sample']
    sample_path = os.path.join(config['data_dir'], 'Samples', sample, f'{sample}.jpg')
    
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    training_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
    
    training_pipeline = (
        dt.Value(training_image)
        >> dt.Affine(
            rotation=lambda: np.random.uniform(config.get('rotation_range', [-30, 30])[0], config.get('rotation_range', [-30, 30])[1]),
            translation=lambda: np.random.uniform(config.get('translation_range', [-10, 10])[0], config.get('translation_range', [-10, 10])[1], 2),
            mode='constant'
        )
        >> dt.Multiply(lambda: np.random.uniform(config['mul_min'], config['mul_max']))
        >> dt.Add(lambda: np.random.uniform(config['add_min'], config['add_max']))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return training_pipeline


def create_multi_particle_pipeline(config):
    """Create training pipeline with multiple particle types"""
    
    particle_types = config.get('particle_types', ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod'])
    samples_dir = os.path.join(config['data_dir'], 'Samples')
    
    # Load all sample images
    sample_images = {}
    for particle_type in particle_types:
        sample_path = os.path.join(samples_dir, particle_type, f'{particle_type}.jpg')
        if os.path.exists(sample_path):
            sample_images[particle_type] = np.array(
                dt.LoadImage(sample_path).resolve()
            ).astype(np.float32)
            print(f"Loaded {particle_type} sample: {sample_images[particle_type].shape}")
        else:
            print(f"Warning: {particle_type} sample not found at {sample_path}")
    
    def get_random_sample():
        """Randomly select a particle type and return its sample"""
        available_types = list(sample_images.keys())
        if not available_types:
            raise ValueError("No sample images found!")
        
        particle_type = random.choice(available_types)
        return sample_images[particle_type]
    
    # Create training pipeline
    training_pipeline = (
        dt.Value(get_random_sample)
        >> dt.Affine(
            rotation=lambda: np.random.uniform(config.get('rotation_range', [-30, 30])[0], config.get('rotation_range', [-30, 30])[1]),
            translation=lambda: np.random.uniform(config.get('translation_range', [-10, 10])[0], config.get('translation_range', [-10, 10])[1], 2),
            mode='constant'
        )
        >> dt.Multiply(lambda: np.random.uniform(config['mul_min'], config['mul_max']))
        >> dt.Add(lambda: np.random.uniform(config['add_min'], config['add_max']))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )
    
    return training_pipeline, sample_images


def create_validation_pipeline(config, sample_images=None):
    """Create validation pipeline with different parameters from training"""
    
    # Get validation-specific augmentation parameters
    val_mul_min = config.get('val_mul_min', config['mul_min'])
    val_mul_max = config.get('val_mul_max', config['mul_max'])
    val_add_min = config.get('val_add_min', config['add_min'])
    val_add_max = config.get('val_add_max', config['add_max'])
    
    if sample_images is None:
        # Single particle validation
        sample = config['sample']
        sample_path = os.path.join(config['data_dir'], 'Samples', sample, f'{sample}.jpg')
        
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample not found: {sample_path}")
        
        validation_image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
        
        validation_pipeline = (
            dt.Value(validation_image)
            >> dt.Affine(
                rotation=lambda: np.random.uniform(config.get('val_rotation_range', [-15, 15])[0], config.get('val_rotation_range', [-15, 15])[1]),
                translation=lambda: np.random.uniform(config.get('val_translation_range', [-5, 5])[0], config.get('val_translation_range', [-5, 5])[1], 2),
                mode='constant'
            )
            >> dt.Multiply(lambda: np.random.uniform(val_mul_min, val_mul_max))
            >> dt.Add(lambda: np.random.uniform(val_add_min, val_add_max))
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )
    else:
        # Multi-particle validation
        def get_validation_sample():
            """Get validation sample with configurable augmentation parameters"""
            available_types = list(sample_images.keys())
            if not available_types:
                raise ValueError("No sample images found!")
            
            particle_type = random.choice(available_types)
            return sample_images[particle_type]
        
        # Use configurable augmentation ranges for validation
        validation_pipeline = (
            dt.Value(get_validation_sample)
            >> dt.Affine(
                rotation=lambda: np.random.uniform(config.get('val_rotation_range', [-15, 15])[0], config.get('val_rotation_range', [-15, 15])[1]),
                translation=lambda: np.random.uniform(config.get('val_translation_range', [-5, 5])[0], config.get('val_translation_range', [-5, 5])[1], 2),
                mode='constant'
            )
            >> dt.Multiply(lambda: np.random.uniform(val_mul_min, val_mul_max))
            >> dt.Add(lambda: np.random.uniform(val_add_min, val_add_max))
            >> dt.MoveAxis(-1, 0)
            >> dt.pytorch.ToTensor(dtype=torch.float32)
        )
    
    return validation_pipeline


def main():
    dir_base = '~/MONA_LodeSTAR/'
    dir_export = dir_base + 'src/lightning_logs/'
    wandb_log_dir = dir_base + "wandb_logs"
    
    # Load configuration
    config = utils.load_yaml('config.yaml')
    
    # Check if multi-particle config exists
    multi_config_path = 'config_multi_particle.yaml'
    if os.path.exists(multi_config_path):
        multi_config = utils.load_yaml(multi_config_path)
        # Merge configurations
        config.update(multi_config)
    
    # Set random seed
    L.seed_everything(config['seed'])
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config=config,
        dir=wandb_log_dir
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        log_model=True,
        dir=wandb_log_dir
    )
    
    # Setup checkpoint directory
    dir_checkpoint = os.path.join(dir_export, wandb.run.id, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='train_within_image_disagreement',
            mode='min',
            filename='{epoch}-{step}-{val_loss:.2f}',
            save_top_k=3,
            every_n_epochs=1,
            dirpath=dir_checkpoint
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Determine training mode
    training_mode = config.get('training_mode', 'single_particle')
    
    if training_mode == 'multi_particle':
        print("Creating multi-particle training pipeline...")
        training_pipeline, sample_images = create_multi_particle_pipeline(config)
        validation_pipeline = create_validation_pipeline(config, sample_images)
        
        # Create enhanced model
        lodestar = MultiParticleLodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr']),
            particle_types=list(sample_images.keys())
        ).build()
        
        # Log particle types
        particle_types = list(sample_images.keys())
        print(f"Training on particle types: {particle_types}")
        wandb.log({"particle_types": particle_types})
        
    else:
        print("Creating single-particle training pipeline...")
        training_pipeline = create_single_particle_pipeline(config)
        validation_pipeline = create_validation_pipeline(config)
        
        # Create standard model
        lodestar = dl.LodeSTAR(
            n_transforms=config['n_transforms'], 
            optimizer=dl.Adam(lr=config['lr'])
        ).build()
        
        print(f"Training on particle type: {config['sample']}")
        wandb.log({"particle_type": config['sample']})
    
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
    
    print(f"Created training dataset with {config['length']} samples")
    print(f"Created validation dataset with {validation_length} samples")
    
    # Log augmentation parameters
    print(f"Training augmentation: mul({config['mul_min']:.2f}, {config['mul_max']:.2f}), add({config['add_min']:.2f}, {config['add_max']:.2f})")
    val_mul_min = config.get('val_mul_min', config['mul_min'])
    val_mul_max = config.get('val_mul_max', config['mul_max'])
    val_add_min = config.get('val_add_min', config['add_min'])
    val_add_max = config.get('val_add_max', config['add_max'])
    print(f"Validation augmentation: mul({val_mul_min:.2f}, {val_mul_max:.2f}), add({val_add_min:.2f}, {val_add_max:.2f})")
    
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
    
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
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
    
    # Save configuration
    utils.save_yaml(config, os.path.join(dir_checkpoint, 'config.yaml'))
    
    # Run training
    print(f"Starting {training_mode} LodeSTAR training...")
    trainer.fit(lodestar, train_dataloader, val_dataloader)
    
    # Save model weights
    torch.save(lodestar.state_dict(), os.path.join(dir_checkpoint, "lodestar_weights.pth"))
    
    # Save final checkpoint
    trainer.save_checkpoint(os.path.join(dir_checkpoint, 'final_epoch.ckpt'))
    
    print(f"Training complete! Model saved to {dir_checkpoint}")
    wandb.finish()


if __name__ == '__main__':
    main() 