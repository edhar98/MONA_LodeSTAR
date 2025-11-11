import torch.nn as nn
import deeptrack.deeplay as dl
from deeplay.components import ConvolutionalNeuralNetwork


class customLodeSTAR(dl.LodeSTAR):
    """LodeSTAR implementation that matches the paper's architecture exactly"""
    
    def __init__(self, n_transforms, optimizer, **kwargs):
        paper_model = ConvolutionalNeuralNetwork(
            in_channels=None,  # Will be inferred from data 
            hidden_channels=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],  # 12 hidden layers
            out_channels=3,  # Δx, Δy, ρ
            **kwargs
        )
        # Customize the architecture to match paper specifications
        # Layer 0-2: 3x Conv2D (3x3, 32) + ReLU
        paper_model.blocks[0].layer.kernel_size = (3, 3)
        paper_model.blocks[0].layer.padding = (1, 1)
        
        # Layer 3: MaxPool2D (2x2) - add after block 2
        paper_model.blocks[2].pooled()
        
        # Layer 4-11: 8x Conv2D (3x3, 32) + ReLU
        for i in range(3, 11):
            paper_model.blocks[i].layer.kernel_size = (3, 3)
            paper_model.blocks[i].layer.padding = (1, 1)
        
        # Layer 12: Conv2D (1x1, 3) - final layer
        paper_model.blocks[12].layer.kernel_size = (1, 1)
        paper_model.blocks[12].layer.padding = (0, 0)
        
        # Initialize weights
        self._initialize_weights(paper_model)
        
        # Call parent constructor with our custom model
        super().__init__(
            model=paper_model,
            num_outputs=2,  # Δx, Δy
            n_transforms=n_transforms,
            optimizer=optimizer,
            **kwargs
        )
    
    def _initialize_weights(self, model):
        """Initialize weights according to paper specifications"""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
