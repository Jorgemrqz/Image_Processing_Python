import argparse
from time import perf_counter
from typing import List
from PIL import Image
import math

OFFSET_DEFAULT = 128

def clamp_i(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def to_grayscale_u8_to_f32(img: Image.Image) -> (List[float], int, int):
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
    w, h = img.size
    px = img.load()
    gray = [0.0] * (w * h)
    if img.mode == "L":
        for y in range(h):
            for x in range(w):
                gray[y*w + x] = float(px[x, y])
    else:
        for y in range(h):
            for x in range(w):
                p = px[x, y]
                r, g, b = (p[0], p[1], p[2])
                gray[y*w + x] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray, w, h

def main():
    ap = argparse.ArgumentParser(description="Emboss kxk (GPU con PyCUDA + Pillow).")
    ap.add_argument("input", help="ruta de la imagen de entrada")
    ap.add_argument("--output", default=None, help="ruta de salida (auto si no se especifica)")
    ap.add_argument("--mask", type=int, default=0, help="tamaño impar del kernel (si 0, se pedirá por teclado)")
    ap.add_argument("--offset", type=int, default=OFFSET_DEFAULT, help="offset a sumar (default 128)")
    args = ap.parse_args()

    img = Image.open(args.input)
    gray, w, h = to_grayscale_u8_to_f32(img)

if __name__ == "__main__":
    main()
