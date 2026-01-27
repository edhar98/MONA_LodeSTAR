# Test Directory

This directory contains development tests for the MONA LodeSTAR project.

## Structure

- **`unit/`** - Unit tests for individual components
- **`regression/`** - Regression tests to ensure no functionality breaks
- **`integration/`** - Integration tests for full workflows

## Running Tests

### Run all tests
```bash
python test/run_tests.py
```

### Run specific test types
```bash
python test/run_tests.py --type unit
python test/run_tests.py --type regression
python test/run_tests.py --type integration
```

### Verbose output
```bash
python test/run_tests.py --verbose
```

## Test Categories

### Unit Tests (`unit/`)
Test individual functions, classes, and modules in isolation:
- `test_lodestar_models.py` - Test LodeSTAR model implementations
- `test_utils.py` - Test utility functions
- `test_skip_connections.py` - Test skip connections implementation

### Regression Tests (`regression/`)
Test that existing functionality continues to work after changes:
- `test_backwards_compatibility.py` - Test backwards compatibility

### Integration Tests (`integration/`)
Test complete workflows and system integration:
- (To be added as needed)

## Writing Tests

### Unit Test Example
```python
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from your_module import your_function

class TestYourModule(unittest.TestCase):
    def test_your_function(self):
        result = your_function(input_data)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
```

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

## Notes

- Model evaluation tests (testing trained models) remain in `src/`
- This directory is for testing the code itself, not model performance
- All tests should be deterministic and not require external data
