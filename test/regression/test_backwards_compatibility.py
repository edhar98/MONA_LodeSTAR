"""
Regression tests for MONA LodeSTAR.

Tests to ensure existing functionality continues to work after changes.
"""

import unittest
import sys
import os
import torch
import numpy as np
import yaml
import tempfile

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from image_generator import generateImage, Object
import utils


class TestRegression(unittest.TestCase):
    """Regression tests to ensure no functionality breaks."""
    
    def test_image_generation_consistency(self):
        """Test that image generation produces consistent results."""
        # Test with fixed seed for reproducibility
        np.random.seed(42)
        
        particle = Object(x=32, y=32, label='Spot', parameters=[[1], [3]])
        bboxes, labels, image, snr = generateImage([particle], 64, 64, 10, [0.8, 1.0])
        
        # Basic checks
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 'Spot')
        self.assertEqual(image.shape, (64, 64))
        self.assertGreater(snr, 0)
    
    def test_config_loading_backwards_compatibility(self):
        """Test that config loading works with existing configs."""
        # Test with minimal config
        minimal_config = {
            'wandb': {'project': 'test'},
            'data_dir': 'data/',
            'lodestar_version': 'default',
            'samples': ['Spot'],
            'length': 100,
            'max_epochs': 1,
            'batch_size': 2,
            'lr': 0.001,
            'n_transforms': 4,
            'devices': 1,
            'lightning': {'accelerator': 'cpu'}
        }
        
        # Save and reload config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name
        
        try:
            loaded_config = utils.load_yaml(config_path)
            self.assertEqual(loaded_config, minimal_config)
        finally:
            os.unlink(config_path)
    
    def test_model_forward_pass_consistency(self):
        """Test that model forward passes produce consistent output shapes."""
        from custom_lodestar import customLodeSTAR
        import deeptrack.deeplay as dl
        
        model = customLodeSTAR(
            n_transforms=4,
            optimizer=dl.Adam(lr=0.0001)
        ).build()
        
        # Test with different input sizes
        test_sizes = [(1, 1, 32, 32), (1, 1, 64, 64), (1, 1, 128, 128)]
        
        for size in test_sizes:
            with torch.no_grad():
                dummy_input = torch.randn(size)
                output = model(dummy_input)
                
                # Check output shape consistency
                expected_channels = 3
                expected_height = size[2] // 2  # Due to pooling
                expected_width = size[3] // 2   # Due to pooling
                
                self.assertEqual(output.shape, (size[0], expected_channels, expected_height, expected_width))


if __name__ == '__main__':
    unittest.main()
