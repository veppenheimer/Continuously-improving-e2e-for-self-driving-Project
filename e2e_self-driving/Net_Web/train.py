#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
TARGET = REPO_ROOT / 'e2e_self-driving' / 'Net' / 'train.py'
if str(TARGET.parent) not in sys.path:
    sys.path.insert(0, str(TARGET.parent))

if __name__ == '__main__':
    runpy.run_path(str(TARGET), run_name='__main__')
