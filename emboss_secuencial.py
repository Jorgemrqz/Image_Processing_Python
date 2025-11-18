import argparse
from time import perf_counter
from typing import List
from PIL import Image
import math

OFFSET_DEFAULT = 128  

def clamp_i(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def to_grayscale_u8_to_f32(img: Image.Image) -> (List[float], int, int):
    """ Devuelve un buffer plano float32 (lista) de tamaño w*h a partir de RGB/RGBA/L """
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
                # luma aproximada ITU-R BT.601
                gray[y*w + x] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray, w, h

def generate_emboss_kernel(k: int) -> List[float]:
    """
    Kernel k×k estilo C++:
      -1 en valores por encima de la diagonal principal,
      +1 por debajo, 0 en la diagonal.
    Escala por 1/sqrt(nonZero).
    """
    K = [0.0] * (k * k)
    diag = k - 1
    non_zero = 0
    for i in range(k):
        for j in range(k):
            s = i + j
            idx = i * k + j
            if s < diag:
                K[idx] = -1.0
                non_zero += 1
            elif s > diag:
                K[idx] = +1.0
                non_zero += 1
            else:
                K[idx] = 0.0
    if non_zero > 0:
        scale = 1.0 / math.sqrt(non_zero)
        for i in range(k * k):
            K[i] *= scale
    return K

def convolve_gray(inbuf: List[float], w: int, h: int, K: List[float], ksize: int) -> List[float]:
    """ Convolución 2D directa (k×k) con bordes clamp. """
    r = ksize // 2
    out = [0.0] * (w * h)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(-r, r + 1):
                yy = clamp_i(y + ky, 0, h - 1)
                krow = (ky + r) * ksize
                row_off = yy * w
                for kx in range(-r, r + 1):
                    xx = clamp_i(x + kx, 0, w - 1)
                    acc += inbuf[row_off + xx] * K[krow + (kx + r)]
            out[y*w + x] = acc
    return out

def save_gray_u8(path: str, src_f32: List[float], w: int, h: int, offset: int, quality: int = 95):
    # convierte a L (8-bit), suma offset y satura
    out = Image.new("L", (w, h))
    pix = out.load()
    i = 0
    for y in range(h):
        for x in range(w):
            iv = int(round(src_f32[i])) + offset
            iv = 0 if iv < 0 else 255 if iv > 255 else iv
            pix[x, y] = iv
            i += 1
    # JPG o PNG según extensión
    if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
        out.save(path, quality=quality, subsampling=0)
    else:
        out.save(path)

def main():
    ap = argparse.ArgumentParser(description="Emboss kxk (Python puro + Pillow, sin NumPy).")
    ap.add_argument("input", help="ruta de la imagen de entrada")
    ap.add_argument("--output", default=None, help="ruta de salida (auto si no se especifica)")
    ap.add_argument("--mask", type=int, default=0, help="tamaño impar del kernel (si 0, se pedirá por teclado)")
    ap.add_argument("--offset", type=int, default=OFFSET_DEFAULT, help="offset a sumar (default 128)")
    args = ap.parse_args()

    # Leer imagen (fuera de la medición)
    img = Image.open(args.input)
    gray, w, h = to_grayscale_u8_to_f32(img)

    # Pedir máscara por teclado si no se pasó por parámetro
    ksize = args.mask
    if ksize <= 0:
        try:
            ksize = int(input("Ingrese tamaño IMPAR del kernel (ej. 3, 9, 21, 65): ").strip())
        except Exception:
            print("Entrada inválida.")
            return
    if ksize < 3 or (ksize % 2 == 0):
        print("Error: el tamaño debe ser impar y >= 3.")
        return

    K = generate_emboss_kernel(ksize)

    t0 = perf_counter()
    out_f32 = convolve_gray(gray, w, h, K, ksize)
    t1 = perf_counter()

    # Guardar (fuera de la medición)
    out_path = args.output or (args.input.rsplit(".", 1)[0] + f"_emboss_k{ksize}.png")
    save_gray_u8(out_path, out_f32, w, h, args.offset, quality=95)

    # Reporte
    print("\n--- RESUMEN (Emboss Python puro) ---")
    print(f"Imagen: {w}x{h}")
    print(f"Kernel: {ksize}x{ksize} (directo, O(K^2))")
    print(f"Offset: {args.offset}")
    print(f"Tiempo SOLO convolución: {t1 - t0:.6f} s")
    print(f"Salida: {out_path}")

if __name__ == "__main__":
    main()
