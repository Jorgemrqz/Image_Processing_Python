import argparse
from time import perf_counter
from typing import List
from PIL import Image

OFFSET_DEFAULT = 128

def main():
    ap = argparse.ArgumentParser(description="Emboss kxk (GPU con PyCUDA + Pillow).")
    ap.add_argument("input", help="ruta de la imagen de entrada")
    ap.add_argument("--output", default=None, help="ruta de salida (auto si no se especifica)")
    ap.add_argument("--mask", type=int, default=0, help="tamaño impar del kernel (si 0, se pedirá por teclado)")
    ap.add_argument("--offset", type=int, default=OFFSET_DEFAULT, help="offset a sumar (default 128)")
    args = ap.parse_args()

    img = Image.open(args.input)

if __name__ == "__main__":
    main()
