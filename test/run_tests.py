#!/usr/bin/env python3
"""
Test runner for MONA LodeSTAR project.

Run all tests or specific test categories.
"""

import unittest
import sys
import os
import argparse


def run_tests(test_type='all', verbose=False):
    """Run tests based on type."""
    
    # Add src to path for imports
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Discover tests based on type
    if test_type == 'all':
        test_dirs = ['unit', 'regression', 'integration']
    else:
        test_dirs = [test_type]
    
    # Find test directory (absolute path)
    test_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_dir_name in test_dirs:
        test_path = os.path.join(test_dir, test_dir_name)
        if os.path.exists(test_path):
            # Use absolute path and specify top_level_dir
            discovered_tests = loader.discover(
                start_dir=test_path, 
                pattern='test_*.py',
                top_level_dir=os.path.dirname(test_dir)  # Project root
            )
            suite.addTest(discovered_tests)
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run MONA LodeSTAR tests')
    parser.add_argument('--type', choices=['all', 'unit', 'regression', 'integration'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Running {args.type} tests...")
    success = run_tests(args.type, args.verbose)
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
