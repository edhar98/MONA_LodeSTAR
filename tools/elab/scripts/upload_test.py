#!/usr/bin/env python3
"""
ELAB utility script for uploading test results.

This script provides a convenient way to upload test results to ELAB.
"""

import sys
import os
import argparse
from datetime import datetime

# Add tools directory to path
tools_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, tools_dir)

from elab.cli.elab_cli_simple import cmd_upload_test_results


def main():
    """Upload test results to ELAB."""
    parser = argparse.ArgumentParser(
        description='Upload test results to ELAB',
        prog='upload-test'
    )
    
    parser.add_argument('--label', type=str, default=None,
                       help='Custom label for the test run (default: timestamp)')
    parser.add_argument('--title-prefix', type=str, default='LodeSTAR Test Results',
                       help='Prefix for the experiment title')
    parser.add_argument('--category', type=int, default=5,
                       help='Experiment category ID (default: 5 for "Full Run")')
    parser.add_argument('--team', type=int, default=1,
                       help='Team ID (default: 1 for "Molecular Nanophotonics Group")')
    parser.add_argument('--experiments', type=int, nargs='*', default=None,
                       help='List of experiment IDs to link')
    parser.add_argument('--items', type=int, nargs='*', default=None,
                       help='List of item IDs to link')
    
    args = parser.parse_args()
    
    print(f"Uploading test results to ELAB...")
    print(f"Label: {args.label or datetime.now().strftime('%Y%m%d-%H%M%S')}")
    print(f"Title prefix: {args.title_prefix}")
    print(f"Category: {args.category}")
    print(f"Team: {args.team}")
    
    return cmd_upload_test_results(
        run_label=args.label,
        title_prefix=args.title_prefix,
        category=args.category,
        team=args.team,
        experiment_links=args.experiments,
        item_links=args.items
    )


if __name__ == '__main__':
    sys.exit(main())
