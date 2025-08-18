import torch
import torch.nn as nn
import deeptrack.deeplay as dl


class customLodeSTAR(dl.LodeSTAR):
    """LodeSTAR implementation that matches the paper's architecture exactly"""
    
    def __init__(self, n_transforms, optimizer, **kwargs):
        # Create the paper-specified model architecture
        paper_model = nn.Sequential(
            # Layer 0-2: 3x Conv2D (3x3, 32) + ReLU
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Layer 3: MaxPool2D (2x2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 4-11: 8x Conv2D (3x3, 32) + ReLU
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Layer 12: Conv2D (1x1, 3) - outputs: Δx, Δy, ρ
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        )
        
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
