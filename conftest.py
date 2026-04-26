"""
Root conftest.py — makes the project root importable so `src.*` imports work
in all test files without needing to install the package.
"""

import sys
import os

# Ensure project root is on the path when running pytest from any directory
sys.path.insert(0, os.path.dirname(__file__))
