"""
Unit tests for utility functions.

Tests utility functions and helper modules.
"""

import unittest
import sys
import os
import tempfile
import yaml

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import utils


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_yaml_load_save(self):
        """Test YAML loading and saving."""
        test_data = {
            'test_key': 'test_value',
            'nested': {
                'key1': 123,
                'key2': [1, 2, 3]
            }
        }
        
        # Test saving
        yaml_path = os.path.join(self.temp_dir, 'test.yaml')
        utils.save_yaml(test_data, yaml_path)
        
        # Test loading
        loaded_data = utils.load_yaml(yaml_path)
        
        self.assertEqual(loaded_data, test_data)
    
    def test_xml_writer(self):
        """Test XML writer functionality."""
        xml_path = os.path.join(self.temp_dir, 'test.xml')
        
        writer = utils.XMLWriter(xml_path, width=100, height=100)
        writer.addObject('TestObject', 10, 20, 30, 40, orientation=0.5)
        writer.setSNR(15.5)
        writer.save(xml_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(xml_path))
        
        # Check file content
        with open(xml_path, 'r') as f:
            content = f.read()
            self.assertIn('TestObject', content)
            self.assertIn('15.5', content)
    
    def test_logger_setup(self):
        """Test logger setup."""
        logger = utils.setup_logger('test_logger')
        
        self.assertEqual(logger.name, 'test_logger')
        self.assertEqual(len(logger.handlers), 1)  # Should have console handler


if __name__ == '__main__':
    unittest.main()
