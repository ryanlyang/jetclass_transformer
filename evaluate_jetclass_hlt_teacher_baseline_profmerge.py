#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thin launcher so `jetclass_transformer` jobs can run the prof-merge experiment
implemented in ATLAS-top-tagging-open-data.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _import_impl():
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "ATLAS-top-tagging-open-data",
        Path("/home/ryreu/atlas/PracticeTagging/ATLAS-top-tagging-open-data"),
        Path("/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data"),
        Path("/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data"),
    ]
    for c in candidates:
        if (c / "evaluate_jetclass_hlt_teacher_baseline_profmerge.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return importlib.import_module("evaluate_jetclass_hlt_teacher_baseline_profmerge")
    raise FileNotFoundError(
        "Could not locate evaluate_jetclass_hlt_teacher_baseline_profmerge.py in known ATLAS-top-tagging-open-data paths."
    )


def main() -> None:
    mod = _import_impl()
    mod.main()


if __name__ == "__main__":
    main()

