#!/usr/bin/env python3
"""Wrapper launcher for the canonical Conan Gradio demo."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.conan_gradio.app import main


if __name__ == '__main__':
    main()
