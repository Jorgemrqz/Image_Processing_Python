import argparse
from time import perf_counter
from typing import List, Tuple, Optional
from PIL import Image

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
