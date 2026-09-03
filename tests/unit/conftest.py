import os
import sys

# Make sure the project root (parent of tests/) is on sys.path
# so that `from src.xxx import yyy` works when running pytest
# from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
