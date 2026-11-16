"""Rebuild this dataset analysis; no training."""
import runpy,sys
from pathlib import Path
script=Path(__file__).resolve().parents[3]/"comparisons/analysis/build.py"
sys.path.insert(0,str(script.parent))
sys.argv=[str(script),"--dataset","fashion"]
runpy.run_path(str(script),run_name="__main__")
