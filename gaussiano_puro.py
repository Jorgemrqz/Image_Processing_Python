import argparse
from time import perf_counter
from typing import List, Tuple, Optional
from PIL import Image

import math

def gaussian_kernel_1d(mask_size: int, sigma: Optional[float] = None) -> List[float]:
    if mask_size < 3:
        mask_size = 3
    if mask_size % 2 == 0:
        mask_size += 1
    radius = (mask_size - 1) // 2
    if sigma is None or sigma <= 0:
        sigma = (radius / 3.0) if radius > 0 else 0.8

    two_sigma2 = 2.0 * sigma * sigma
    vals = []
    s = 0.0
    for i in range(-radius, radius + 1):
        v = math.exp(-(i * i) / two_sigma2)
        vals.append(v)
        s += v
    return [v / s for v in vals]

def clamp(v: int, lo: int, hi: int) -> int:
    if v < lo: return lo
    if v > hi: return hi
    return v

def convolve_horizontal(src: List[List[float]], w: int, h: int, k: List[float]) -> List[List[float]]:
    kr = len(k) // 2
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        row = src[y]
        dst_row = out[y]
        for x in range(w):
            acc = 0.0
            for t in range(-kr, kr + 1):
                xx = clamp(x + t, 0, w - 1)
                acc += row[xx] * k[t + kr]
            dst_row[x] = acc
    return out

def convolve_vertical(src: List[List[float]], w: int, h: int, k: List[float]) -> List[List[float]]:
    kr = len(k) // 2
    out = [[0.0] * w for _ in range(h)]
    for x in range(w):
        for y in range(h):
            acc = 0.0
            for t in range(-kr, kr + 1):
                yy = clamp(y + t, 0, h - 1)
                acc += src[yy][x] * k[t + kr]
            out[y][x] = acc
    return out

def main():
    ap = argparse.ArgumentParser(description="Gaussian Blur separable (puro Python + Pillow, sin NumPy).")
    ap.add_argument("input", help="Ruta de la imagen de entrada")
    ap.add_argument("--mask", type=int, default=7, help="Tamaño de máscara impar (>=3)")
    ap.add_argument("--sigma", type=float, default=-1.0, help="Sigma (<=0 deriva sigma=radius/3)")
    ap.add_argument("--output", default=None, help="Ruta de salida")
    args = ap.parse_args()

    # TODO: implementar filtro gaussiano

if __name__ == "__main__":
    main()
