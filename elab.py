#!/usr/bin/env python3
"""
Convenience script for ELAB operations from project root.

This script provides easy access to ELAB tools from the project root directory.
"""

import sys
import os
import subprocess

def main():
    """Run ELAB tools from project root."""
    if len(sys.argv) < 2:
        print("Usage: python elab.py <command> [args...]")
        print("\nAvailable commands:")
        print("  upload-training  - Upload training results")
        print("  upload-test      - Upload test results")
        print("  cli-full         - Run full ELAB CLI")
        print("  cli-simple       - Run simple ELAB CLI")
        print("\nExamples:")
        print("  python elab.py upload-training")
        print("  python elab.py upload-test --label experiment_001")
        print("  python elab.py cli-simple upload-training --help")
        return 1
    
    command = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Map commands to actual scripts
    command_map = {
        'upload-training': 'tools/elab/scripts/upload_training.py',
        'upload-test': 'tools/elab/scripts/upload_test.py',
        'cli-full': 'tools/elab/cli/elab_cli.py',
        'cli-simple': 'tools/elab/cli/elab_cli_simple.py'
    }
    
    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands:", list(command_map.keys()))
        return 1
    
    script_path = command_map[command]
    
    # Run the script with remaining arguments
    try:
        result = subprocess.run([sys.executable, script_path] + args)
        return result.returncode
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
