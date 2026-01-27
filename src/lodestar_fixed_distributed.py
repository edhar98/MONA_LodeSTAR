#!/usr/bin/env python3
"""
Fixed LodeSTAR implementation with proper distributed training support
"""

import deeptrack.deeplay as dl
import torch
from typing import Literal
import torchmetrics as tm


class LodeSTARFixedDistributed(dl.LodeSTAR):
    """
    LodeSTAR with fixed distributed training logging
    """
    
    def training_step(self, batch, batch_idx):
        """Training step with proper distributed logging"""
        x, y = self.train_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if not isinstance(loss, dict):
            loss = {"loss": loss}

        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,  # Fix for distributed training
            )

        self.log_metrics(
            "train", y_hat, y, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return sum(loss.values())
    
    def log_metrics(
        self, kind: Literal["train", "val", "test"], y_hat, y, **logger_kwargs
    ):
        """Log metrics with proper distributed training support"""
        ys = self.metrics_preprocess(y_hat, y)

        metrics: tm.MetricCollection = getattr(self, f"{kind}_metrics")
        metrics(*ys)

        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                sync_dist=True,  # Fix for distributed training
                **logger_kwargs,
            )
