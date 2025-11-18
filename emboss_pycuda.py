import argparse
from time import perf_counter
from typing import List
from PIL import Image
import math

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


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

def generate_emboss_kernel(k: int) -> List[float]:
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

CUDA_SRC = r"""
extern "C"
__global__ void convolve_gray_gpu(
    const float *inbuf,
    float *outbuf,
    const float *K,
    int w,
    int h,
    int ksize
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int r = ksize / 2;
    float acc = 0.0f;

    for (int ky = -r; ky <= r; ++ky) {
        int yy = y + ky;
        if (yy < 0) yy = 0;
        else if (yy >= h) yy = h - 1;

        int krow = (ky + r) * ksize;
        int row_off = yy * w;

        for (int kx = -r; kx <= r; ++kx) {
            int xx = x + kx;
            if (xx < 0) xx = 0;
            else if (xx >= w) xx = w - 1;

            float val = inbuf[row_off + xx];
            float kval = K[krow + (kx + r)];
            acc += val * kval;
        }
    }

    outbuf[y * w + x] = acc;
}
""";

mod = SourceModule(CUDA_SRC)
convolve_gray_gpu_kernel = mod.get_function("convolve_gray_gpu")

def convolve_gray_gpu(gray: List[float], w: int, h: int,
                      K: List[float], ksize: int):
    """Convolución emboss ejecutada completamente en GPU."""
    gray_np = np.asarray(gray, dtype=np.float32)
    K_np = np.asarray(K, dtype=np.float32)

    d_in = cuda.mem_alloc(gray_np.nbytes)
    d_out = cuda.mem_alloc(gray_np.nbytes)
    d_K = cuda.mem_alloc(K_np.nbytes)

    cuda.memcpy_htod(d_in, gray_np)
    cuda.memcpy_htod(d_K, K_np)

    block_x, block_y = 16, 16
    grid_x = (w + block_x - 1) // block_x
    grid_y = (h + block_y - 1) // block_y

    start_evt = cuda.Event()
    end_evt = cuda.Event()

    start_evt.record()

    convolve_gray_gpu_kernel(
        d_in,
        d_out,
        d_K,
        np.int32(w),
        np.int32(h),
        np.int32(ksize),
        block=(block_x, block_y, 1),
        grid=(grid_x, grid_y, 1),
    )

    end_evt.record()
    end_evt.synchronize()

    kernel_ms = start_evt.time_till(end_evt)

    out_np = np.empty_like(gray_np)
    cuda.memcpy_dtoh(out_np, d_out)

    d_in.free()
    d_out.free()
    d_K.free()

    return out_np.tolist(), kernel_ms / 1000.0

def main():
    ap = argparse.ArgumentParser(description="Emboss kxk (GPU con PyCUDA + Pillow).")
    ap.add_argument("input", help="ruta de la imagen de entrada")
    ap.add_argument("--output", default=None, help="ruta de salida (auto si no se especifica)")
    ap.add_argument("--mask", type=int, default=0, help="tamaño impar del kernel (si 0, se pedirá por teclado)")
    ap.add_argument("--offset", type=int, default=OFFSET_DEFAULT, help="offset a sumar (default 128)")
    args = ap.parse_args()

    img = Image.open(args.input)
    gray, w, h = to_grayscale_u8_to_f32(img)

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

if __name__ == "__main__":
    main()
