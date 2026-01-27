import torch
import torch.nn as nn
import deeptrack.deeplay as dl
from deeplay.components import ConvolutionalNeuralNetwork


class LodeSTARWithSkipConnections(dl.LodeSTAR):
    """LodeSTAR implementation with skip connections for improved spatial resolution"""
    
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
        
        # Create skip connection layers
        self._create_skip_connections()
        
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
    
    def _create_skip_connections(self):
        """Create skip connection layers for channel matching and upsampling"""
        
        # Skip connection 1: Block 0 (32) → Block 4 (64)
        self.skip_conv_0_to_4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.skip_upsample_0_to_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Skip connection 2: Block 1 (32) → Block 5 (64)  
        self.skip_conv_1_to_5 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.skip_upsample_1_to_5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Skip connection 3: Block 2 (64) → Block 6 (64) - no channel change needed
        self.skip_upsample_2_to_6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Skip connection 4: Block 3 (64) → Block 7 (64) - no channel change needed
        # No upsampling needed as spatial dimensions are the same
        
        # Skip connection 5: Block 4 (64) → Block 8 (64) - no channel change needed
        # No upsampling needed as spatial dimensions are the same
    
    def forward(self, x):
        """Forward pass with skip connections"""
        
        # Store intermediate activations for skip connections
        activations = {}
        
        # Forward through blocks 0-1 (32 channels each)
        x0 = self.base_model.blocks[0](x)
        activations[0] = x0
        
        x1 = self.base_model.blocks[1](x0)
        activations[1] = x1
        
        # Forward through block 2 (64 channels, with pooling)
        x2 = self.base_model.blocks[2](x1)
        activations[2] = x2
        
        # Forward through block 3 (64 channels)
        x3 = self.base_model.blocks[3](x2)
        activations[3] = x3
        
        # Forward through block 4 (64 channels) + Skip from block 0
        x4_base = self.base_model.blocks[4](x3)
        # Process skip connection: 32->64 channels and upsample to match spatial dimensions
        skip_0_processed = self.skip_conv_0_to_4(activations[0])
        skip_0_upsampled = self.skip_upsample_0_to_4(skip_0_processed)
        # Ensure spatial dimensions match
        if skip_0_upsampled.shape[2:] != x4_base.shape[2:]:
            skip_0_upsampled = torch.nn.functional.interpolate(
                skip_0_upsampled, size=x4_base.shape[2:], mode='bilinear', align_corners=False
            )
        x4 = x4_base + skip_0_upsampled
        activations[4] = x4
        
        # Forward through block 5 (64 channels) + Skip from block 1
        x5_base = self.base_model.blocks[5](x4)
        # Process skip connection: 32->64 channels and upsample to match spatial dimensions
        skip_1_processed = self.skip_conv_1_to_5(activations[1])
        skip_1_upsampled = self.skip_upsample_1_to_5(skip_1_processed)
        # Ensure spatial dimensions match
        if skip_1_upsampled.shape[2:] != x5_base.shape[2:]:
            skip_1_upsampled = torch.nn.functional.interpolate(
                skip_1_upsampled, size=x5_base.shape[2:], mode='bilinear', align_corners=False
            )
        x5 = x5_base + skip_1_upsampled
        activations[5] = x5
        
        # Forward through block 6 (64 channels) + Skip from block 2
        x6_base = self.base_model.blocks[6](x5)
        # Process skip connection: upsample to match spatial dimensions
        skip_2_upsampled = self.skip_upsample_2_to_6(activations[2])
        # Ensure spatial dimensions match
        if skip_2_upsampled.shape[2:] != x6_base.shape[2:]:
            skip_2_upsampled = torch.nn.functional.interpolate(
                skip_2_upsampled, size=x6_base.shape[2:], mode='bilinear', align_corners=False
            )
        x6 = x6_base + skip_2_upsampled
        activations[6] = x6
        
        # Forward through block 7 (64 channels) + Skip from block 3
        x7_base = self.base_model.blocks[7](x6)
        # Direct skip, same spatial dimensions
        x7 = x7_base + activations[3]
        activations[7] = x7
        
        # Forward through block 8 (64 channels) + Skip from block 4
        x8_base = self.base_model.blocks[8](x7)
        # Direct skip, same spatial dimensions
        x8 = x8_base + activations[4]
        
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
        for m in [self.skip_conv_0_to_4, self.skip_conv_1_to_5]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
