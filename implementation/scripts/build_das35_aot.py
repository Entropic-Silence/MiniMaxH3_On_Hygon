"""Run the existing H3 AOT builder with the verified DAS 3.5.1 parser adapter."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'ninetoothed/src'),str(root/'tools')]
import torch
from ifab_triton_compat import install

install()
sys.argv[0] = str(root/'scripts/build_h3_attention.py')
runpy.run_path(sys.argv[0],run_name='__main__')
