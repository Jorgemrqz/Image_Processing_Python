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

def image_to_planes(img: Image.Image) -> Tuple[List[List[float]], List[List[float]], List[List[float]], Optional[List[List[float]]]]:
    w, h = img.size
    pixels = img.load()
    R = [[0.0] * w for _ in range(h)]
    G = [[0.0] * w for _ in range(h)]
    B = [[0.0] * w for _ in range(h)]
    A = [[0.0] * w for _ in range(h)] if img.mode == "RGBA" else None

    for y in range(h):
        for x in range(w):
            px = pixels[x, y]
            if img.mode == "RGBA":
                r, g, b, a = px
                A[y][x] = float(a)
            else:
                r, g, b = px
            R[y][x] = float(r)
            G[y][x] = float(g)
            B[y][x] = float(b)
    return R, G, B, A

def planes_to_image(R: List[List[float]], G: List[List[float]], B: List[List[float]], A: Optional[List[List[float]]]) -> Image.Image:
    h = len(R)
    w = len(R[0]) if h > 0 else 0
    out = Image.new("RGBA" if A is not None else "RGB", (w, h))
    pix = out.load()
    for y in range(h):
        for x in range(w):
            r = int(R[y][x] + 0.5); r = 0 if r < 0 else 255 if r > 255 else r
            g = int(G[y][x] + 0.5); g = 0 if g < 0 else 255 if g > 255 else g
            b = int(B[y][x] + 0.5); b = 0 if b < 0 else 255 if b > 255 else b
            if A is not None:
                a = int(A[y][x] + 0.5); a = 0 if a < 0 else 255 if a > 255 else a
                pix[x, y] = (r, g, b, a)
            else:
                pix[x, y] = (r, g, b)
    return out

def main():
    ap = argparse.ArgumentParser(description="Gaussian Blur separable (puro Python + Pillow, sin NumPy).")
    ap.add_argument("input", help="Ruta de la imagen de entrada")
    ap.add_argument("--mask", type=int, default=7, help="Tamaño de máscara impar (>=3)")
    ap.add_argument("--sigma", type=float, default=-1.0, help="Sigma (<=0 deriva sigma=radius/3)")
    ap.add_argument("--output", default=None, help="Ruta de salida")
    args = ap.parse_args()

    img = Image.open(args.input)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    w, h = img.size
    R, G, B, A = image_to_planes(img)

    sigma = None if args.sigma <= 0 else args.sigma
    k = gaussian_kernel_1d(args.mask, sigma)

    t0 = perf_counter()

    R_h = convolve_horizontal(R, w, h, k)
    R_v = convolve_vertical(R_h, w, h, k)

    G_h = convolve_horizontal(G, w, h, k)
    G_v = convolve_vertical(G_h, w, h, k)

    B_h = convolve_horizontal(B, w, h, k)
    B_v = convolve_vertical(B_h, w, h, k)

    t1 = perf_counter()

    out_img = planes_to_image(R_v, G_v, B_v, A)
    out_path = args.output or (args.input.rsplit(".", 1)[0] + f"_gauss_purepy_k{len(k)}.png")
    out_img.save(out_path)

    print("\n--- RESUMEN (Python puro) ---")
    print(f"Tamaño de imagen: {w}x{h}px")
    print(f"Máscara: {len(k)} x {len(k)} (separable)")
    print(f"Sigma: {sigma if sigma is not None else f'(derivado => {(len(k)-1)//2}/3)'}")
    print(f"Tiempo SOLO CPU (loops Python): {t1 - t0:.6f} s")
    print(f"Salida: {out_path}")

if __name__ == "__main__":
    main()
