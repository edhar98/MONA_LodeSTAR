import torch
import torch.nn as nn
import deeptrack.deeplay as dl
from deeplay.components import ConvolutionalNeuralNetwork


class LodeSTARSimpleSkip(dl.LodeSTAR):
    """Simplified LodeSTAR with minimal skip connections for better stability"""
    
    def __init__(self, n_transforms, optimizer, **kwargs):
        # Create the default LodeSTAR architecture
        default_model = ConvolutionalNeuralNetwork(
            in_channels=None,  # Will be inferred from data 
            hidden_channels=[32, 32, 64, 64, 64, 64, 64, 64, 64],  # 9 layers
            out_channels=3,  # Δx, Δy, ρ
            **kwargs
        )
        
        # Add pooling after block 2 (as in default)
        default_model.blocks[2].pooled()
        
        # Store the base model
        self.base_model = default_model
        
        # Create only essential skip connections (fewer for stability)
        self._create_simple_skip_connections()
        
        # Initialize weights
        self._initialize_weights(default_model)
        
        # Call parent constructor with our enhanced model
        super().__init__(
            model=default_model,
            num_outputs=2,  # Δx, Δy
            n_transforms=n_transforms,
            optimizer=optimizer,
            **kwargs
        )
    
    def _create_simple_skip_connections(self):
        """Create only essential skip connections for stability"""
        
        # Only 2 skip connections instead of 5
        # Skip connection 1: Block 2 (64) → Block 6 (64) - across pooling
        self.skip_conv_2_to_6 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        
        # Skip connection 2: Block 3 (64) → Block 7 (64) - direct skip
        # No additional layers needed for same dimensions
    
    def forward(self, x):
        """Forward pass with minimal skip connections"""
        
        # Store intermediate activations for skip connections
        activations = {}
        
        # Forward through blocks 0-1 (32 channels each)
        x0 = self.base_model.blocks[0](x)
        x1 = self.base_model.blocks[1](x0)
        
        # Forward through block 2 (64 channels, with pooling)
        x2 = self.base_model.blocks[2](x1)
        activations[2] = x2
        
        # Forward through block 3 (64 channels)
        x3 = self.base_model.blocks[3](x2)
        activations[3] = x3
        
        # Forward through block 4 (64 channels)
        x4 = self.base_model.blocks[4](x3)
        
        # Forward through block 5 (64 channels)
        x5 = self.base_model.blocks[5](x4)
        
        # Forward through block 6 (64 channels) + Skip from block 2
        x6_base = self.base_model.blocks[6](x5)
        # Process skip connection: 1x1 conv + direct addition
        skip_2_processed = self.skip_conv_2_to_6(activations[2])
        x6 = x6_base + skip_2_processed
        activations[6] = x6
        
        # Forward through block 7 (64 channels) + Skip from block 3
        x7_base = self.base_model.blocks[7](x6)
        # Direct skip, same spatial dimensions
        x7 = x7_base + activations[3]
        activations[7] = x7
        
        # Forward through block 8 (64 channels)
        x8 = self.base_model.blocks[8](x7)
        
        # Final output layer
        output = self.base_model.blocks[9](x8)
        
        return output
    
    def _initialize_weights(self, model):
        """Initialize weights according to paper specifications"""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize skip connection weights
        if hasattr(self, 'skip_conv_2_to_6'):
            nn.init.kaiming_normal_(self.skip_conv_2_to_6.weight, mode='fan_out', nonlinearity='relu')
            if self.skip_conv_2_to_6.bias is not None:
                nn.init.constant_(self.skip_conv_2_to_6.bias, 0)
