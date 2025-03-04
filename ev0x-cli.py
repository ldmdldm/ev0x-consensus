#!/usr/bin/env python3
"""
ev0x-cli: Command Line Interface for the ev0x Evolutionary Model Consensus System

This script provides a comprehensive interface for managing the ev0x system,
including model management, deployment, monitoring, and Flare ecosystem integration.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.cli.ev0x_cli import main as cli_main
except ImportError:
    print("Error: Could not import ev0x CLI modules.")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)


def main():
    """Entry point for the ev0x CLI tool."""
    parser = argparse.ArgumentParser(
        description="ev0x: Evolutionary Model Consensus Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
deploy        Deploy ev0x to development, staging, or production environments
models        Manage AI models (add, remove, list, update)
monitor       Monitor system performance and health
consensus     Run and evaluate consensus algorithms
evolve        Trigger evolutionary processes on models
integrate     Manage Flare ecosystem integrations
tee           Manage Trusted Execution Environment settings

Examples:
ev0x-cli.py deploy production    # Deploy to production environment
ev0x-cli.py models list          # List all available models
ev0x-cli.py consensus run        # Run the consensus engine
"""
    )

    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Pass control to the main CLI implementation
    return cli_main(sys.argv)


if __name__ == "__main__":
    main()

