"""Pytest configuration for the AORC-Heat-Pipeline test suite.

Adds the project root to ``sys.path`` so the tests can ``import metrics``
regardless of the directory pytest is invoked from.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
