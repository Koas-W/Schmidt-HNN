#!/usr/bin/env python3
"""FPU chain: the w/o-separation variant with the inverse mass matrix represented (M^-1 = L L^T).

Runs scripts/run_fpu.py with the same options, with the mass parameterisation of pshnn_concat
switched to "inverse". Used for the comparison row of the appendix table (tag grid_k4_concat_minv).
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("rfs", ROOT / "scripts/run_fpu.py")
rfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfs)
rfs.MODELS_CFG["pshnn_concat"]["mass_param"] = "inverse"
rfs.main()
