import os
import uuid
from datetime import datetime
from lightning.pytorch.callbacks import Callback

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbLogger = None


class NoOpSummary:
    def __setitem__(self, key, value):
        pass
    
    def __getitem__(self, key):
        return None


class NoOpLogger:
    def __init__(self, save_dir="lightning_logs", **kwargs):
        self._run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._save_dir = save_dir
        self.summary = NoOpSummary()
    
    @property
    def experiment(self):
        return self
    
    @property
    def id(self):
        return self._run_id
    
    @property
    def save_dir(self):
        return self._save_dir
    
    def log_hyperparams(self, params):
        pass
    
    def log_metrics(self, metrics, step=None):
        pass
    
    def finish(self):
        pass


def get_logger(config, particle_type):
    if not WANDB_AVAILABLE or not config.get('wandb', {}).get('enabled', True):
        return NoOpLogger(save_dir=config.get('wandb_log_dir', 'wandb_logs'))
    
    wandb_config = config.get('wandb', {})
    return WandbLogger(
        project=wandb_config.get('project', f"LodeSTAR_{particle_type}"),
        entity=wandb_config.get('entity'),
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        log_model="all",
        save_dir=config.get('wandb_log_dir', 'wandb_logs'),
        config=config
    )


def get_run_id(logger_instance):
    if WANDB_AVAILABLE and isinstance(logger_instance, WandbLogger):
        return logger_instance.experiment.id
    return logger_instance.id if hasattr(logger_instance, 'id') else logger_instance._run_id


def set_summary(key, value):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.summary[key] = value


def log_images(key, images, step=0):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({key: images}, step=step)


def finish_run(logger_instance):
    if WANDB_AVAILABLE and isinstance(logger_instance, WandbLogger):
        logger_instance.experiment.finish()
    elif hasattr(logger_instance, 'finish'):
        logger_instance.finish()


class TrainingMetricsCallback(Callback):
    def __init__(self, particle_type, train_dataloader=None, sample_path=None, app_logger=None):
        super().__init__()
        self.particle_type = particle_type
        self.train_dataloader = train_dataloader
        self.sample_path = sample_path
        self.app_logger = app_logger
        self.setup_done = False
        
    def on_fit_start(self, trainer, pl_module):
        if self.setup_done:
            return
        self.setup_done = True
        
        import torch
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        set_summary("model_params", total_params)
        set_summary("trainable_params", trainable_params)
        
        if self.app_logger:
            self.app_logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        if not WANDB_AVAILABLE or wandb.run is None:
            return
            
        if self.sample_path is not None:
            try:
                import matplotlib.pyplot as plt
                original_img = plt.imread(self.sample_path)
                log_images("original_sample", [wandb.Image(original_img, caption="Original (no augmentation)")], step=0)
            except Exception as e:
                if self.app_logger:
                    self.app_logger.warning(f"Could not log original sample image: {e}")
        
        if self.train_dataloader is not None:
            try:
                import numpy as np
                batch = next(iter(self.train_dataloader))
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                sample_images = []
                for i, img in enumerate(images[:4]):
                    arr = img.squeeze().cpu().numpy()
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                    sample_images.append(wandb.Image(arr, caption=f"Augmented {i}"))
                log_images("augmented_samples", sample_images, step=0)
            except Exception as e:
                if self.app_logger:
                    self.app_logger.warning(f"Could not log augmented sample images: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        import torch
        grad_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        pl_module.log("grad_norm", grad_norm, on_epoch=True, logger=True)
    
    def on_train_end(self, trainer, pl_module):
        import torch
        final_metrics = trainer.callback_metrics
        for key in ['train_within_image_disagreement', 'val_within_image_disagreement']:
            if key in final_metrics:
                val = final_metrics[key]
                set_summary(f"final_{key}", val.item() if isinstance(val, torch.Tensor) else val)
        set_summary("final_epoch", trainer.current_epoch)

