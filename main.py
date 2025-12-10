#!/usr/bin/env python
"""
Hospital Stock Management AI - Main Entry Point
================================================

Clinique du Mont Vert - Stock Prediction System

Usage:
    python main.py predict --product "Poulet Frais" --days 30
    python main.py analyze --dataset enriched
    python main.py list-products
    python main.py list-datasets

For help:
    python main.py --help
    python main.py predict --help
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.cli import main

if __name__ == '__main__':
    sys.exit(main())
