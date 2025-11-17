import argparse
from PIL import Image

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def main():
    ap = argparse.ArgumentParser(
        description="Gaussian blur separable en PyCUDA (sin NumPy, mide SOLO GPU)."
    )
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--mask", type=int, default=21)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--block", type=int, nargs=2, default=[16, 16],
                    help="bloque CUDA (bx by)")
    args = ap.parse_args()

if __name__ == "__main__":
    main()
