"""
Pytest configuration for GeoDelta Backend test suite.
Ensures app package is on sys.path.
"""

import sys
import os

# Insert backend directory into sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
