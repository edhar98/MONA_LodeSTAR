"""
Unit tests for LodeSTAR model implementations.

Tests individual model components and implementations.
"""

import unittest
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from custom_lodestar import customLodeSTAR
import deeptrack.deeplay as dl


class TestLodeSTARImplementations(unittest.TestCase):
    """Test different LodeSTAR implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = (1, 1, 64, 64)
        self.dummy_input = torch.randn(self.input_size)
    
    def test_custom_lodestar_forward_pass(self):
        """Test custom LodeSTAR forward pass."""
        model = customLodeSTAR(
            n_transforms=4,
            optimizer=dl.Adam(lr=0.0001)
        ).build()
        
        with torch.no_grad():
            output = model(self.dummy_input)
        
        # Check output shape: should be (batch, 3, height/2, width/2)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(output.shape, expected_shape)
    
    
    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        models = {
            'default': dl.LodeSTAR(n_transforms=4, optimizer=dl.Adam(lr=0.0001)).build(),
            'custom': customLodeSTAR(n_transforms=4, optimizer=dl.Adam(lr=0.0001)).build(),
        }
        
        for name, model in models.items():
            with torch.no_grad():
                _ = model(self.dummy_input)  # Initialize parameters
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Basic sanity checks
            self.assertGreater(total_params, 1000, f"{name} should have more than 1000 parameters")
            self.assertEqual(total_params, trainable_params, f"{name} should have all parameters trainable")


if __name__ == '__main__':
    unittest.main()
