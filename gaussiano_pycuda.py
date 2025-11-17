import argparse
from PIL import Image

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import math
from array import array

def gaussian_kernel_1d(mask_size: int, sigma: float | None):
    if mask_size < 3:
        mask_size = 3
    if mask_size % 2 == 0:
        mask_size += 1
    r = (mask_size - 1) // 2
    if not sigma or sigma <= 0:
        sigma = (r / 3.0) if r > 0 else 0.8

    two_sigma2 = 2.0 * sigma * sigma
    vals = []
    s = 0.0
    for i in range(-r, r + 1):
        v = math.exp(-(i * i) / two_sigma2)
        vals.append(v)
        s += v
    vals = [v / s for v in vals]      
    return array("f", vals), r        

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
