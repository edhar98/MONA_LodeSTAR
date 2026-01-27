#!/usr/bin/env python3
"""
Main ELAB CLI entry point for MONA LodeSTAR project.

This script provides a unified interface to all ELAB operations.
"""

import sys
import os
import argparse

# Add tools directory to path
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tools_dir)

from elab.cli.elab_cli import main as full_cli_main
from elab.cli.elab_cli_simple import main as simple_cli_main


def main():
    """Main ELAB CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ELAB CLI tools for MONA LodeSTAR project',
        prog='elab-tools'
    )
    
    subparsers = parser.add_subparsers(dest='tool', required=True, help='ELAB tool to use')
    
    # Full CLI tool
    full_parser = subparsers.add_parser(
        'full', 
        help='Full ELAB CLI with all features',
        description='Use the full ELAB CLI with comprehensive features'
    )
    full_parser.set_defaults(func=lambda args: full_cli_main(sys.argv[2:]))
    
    simple_parser = subparsers.add_parser(
        'simple',
        help='Simplified ELAB CLI for common operations',
        description='Use the simplified ELAB CLI for common operations'
    )
    simple_parser.set_defaults(func=lambda args: simple_cli_main(sys.argv[2:]))
    
    args = parser.parse_args()
    
    # Run the selected tool
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error running ELAB tool: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
