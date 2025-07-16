import os
import sys
import torch
import deeptrack as dt
import deeplay as dl 
import numpy as np
import matplotlib.pyplot as plt
from deeplay import LodeSTAR
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import utils


def main():
    dir_base = '/home/edgar/Desktop/mydocs/SHK/MONA/LodeSTAR/'

    dir_export = dir_base + 'src/lightning_logs/'
    # Load configuration
    config = utils.load_yaml('config.yaml')

    # Set random seed
    L.seed_everything(config['seed'])

    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config=config
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        tags=config['wandb']['tags'],
        log_model=True
    )

    # Setup checkpoint directory
    dir_checkpoint = os.path.join(dir_export, wandb.run.id, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor=config['monitor'],
            mode=config['mode'],
            filename='{epoch}-{step}-{val_loss:.2f}',
            save_top_k=3,
            every_n_epochs=1,
            dirpath=dir_checkpoint
        ),
        LearningRateMonitor(logging_interval='epoch'),
        
    ]

    sample = config['sample']
    training_image = np.array(dt.LoadImage(os.path.join(config['data_dir'], 'Samples', sample, sample + '.jpg')).resolve()).astype(np.float32)
    training_pipeline = (
        dt.Value(training_image)
        >> dt.Multiply(lambda: np.random.uniform(config['mul_min'], config['mul_max']))
        >> dt.Add(lambda: np.random.uniform(config['add_min'], config['add_max']))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )

    training_dataset = dt.pytorch.Dataset(training_pipeline, length=config['length'], 
                                        replace=False)
    
    dataloader = dl.DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    lodestar = dl.LodeSTAR(n_transforms=config['n_transforms'], optimizer=dl.Adam(lr=config['lr'])).build()
    
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

    trainer.fit(lodestar, dataloader)
    
    torch.save(lodestar.state_dict(), os.path.join(dir_checkpoint, "lodestar_weights.pth"))

    # Save final checkpoint
    trainer.save_checkpoint(os.path.join(dir_checkpoint, 'final_epoch.ckpt'))
    
    wandb.finish()

if __name__ == '__main__':
    main()