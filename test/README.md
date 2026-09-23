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

### Regression Tests (`regression/`)
Test that existing functionality continues to work after changes:
- `test_backwards_compatibility.py` - Test backwards compatibility

### Integration Tests (`integration/`)

The following checks are opt-in and are not discovered by the `test_*.py` unittest runner. Run from the repository root:

```bash
/opt/mona_jupyterhub_env/bin/python -B test/integration/smoke_hub_pipeline.py
node test/integration/check_web_ui.cjs
```

`smoke_hub_pipeline.py` requires the MONA scientific/web environment, PyTorch, `httpx`, the local `5m4rtzfx` JP_Fe_wf_2_40 model catalogue entry, saved run configuration and weights, dataset-04 PNG frames `JP_Fe_wf_2_40_slm075_574_001.png` through `_003.png`, and orientation template `data/Samples/JP_Fe_wf_2_40/Samples/f000_d003_phi0234.0.png`. It forces CPU execution and checks isolated Hub identity, upload/crop/server input, real-model batch detection, template orientation schema, tracking, an ABP plot, export response/bytes, and a second application lifespan. State/output goes into a temporary directory; no training or external HTTP request occurs. Its POSIX alarm bounds execution to 180 seconds; forced timeout can leave that run's temporary directory behind.

The 2026-09-23 run reported 3 frames, 367 detections, 121 tracks and 356 track rows. Template processing returned 120 finite detections using a 10-degree angle step and 2-pixel search radius. These are smoke observations, not fixed acceptance counts or scientific validation. Three frames cannot validate physical fits; coarse template settings check schema, not orientation accuracy. The test does not cover TDMS, a real browser, or HTTP streaming of FileResponse (it checks the response and file bytes directly).

`check_web_ui.cjs` needs Node.js and only built-in modules. It parses the shipped inline JavaScript and exercises proxy prefixes, interrupted-training recovery, and ABP plot/error states with mocked elements. It does not run a browser or real DOM. Native browser verification remains a separate review gate; neither check proves the installed Hub checkout is deployed or operational.

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

- Model evaluation scripts (testing trained models) live in `src/detection/`
- This directory is for testing the code itself, not model performance
- Default unit/regression tests should be deterministic and not require external data; the opt-in real-model smoke has explicit local fixture requirements above.
