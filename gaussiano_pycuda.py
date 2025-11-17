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

CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ unsigned char sat_uchar(float x) {
    if (x < 0.0f) return 0;
    if (x > 255.0f) return 255;
    return (unsigned char)(x + 0.5f);
}

// HxWx4 (RGBA intercalado). Difumina R,G,B. A se copia.
__global__ void gauss1d_h_rgba(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int W, int H,
    const float* __restrict__ k, int kr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= W || y >= H) return;

    int base = (y * W + x) * 4;

    // R,G,B
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int t = -kr; t <= kr; ++t) {
            int xx = clampi(x + t, 0, W - 1);
            int idx = (y * W + xx) * 4 + c;
            acc += (float)src[idx] * k[t + kr];
        }
        dst[base + c] = sat_uchar(acc);
    }
    // A tal cual
    dst[base + 3] = src[base + 3];
}

__global__ void gauss1d_v_rgba(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int W, int H,
    const float* __restrict__ k, int kr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= W || y >= H) return;

    int base = (y * W + x) * 4;

    // R,G,B
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int t = -kr; t <= kr; ++t) {
            int yy = clampi(y + t, 0, H - 1);
            int idx = (yy * W + x) * 4 + c;
            acc += (float)src[idx] * k[t + kr];
        }
        dst[base + c] = sat_uchar(acc);
    }
    // A tal cual
    dst[base + 3] = src[base + 3];
}

} // extern "C"
"""

def main():
    ap = argparse.ArgumentParser(description="Gaussian blur separable en PyCUDA (sin NumPy, mide SOLO GPU).")
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--mask", type=int, default=21)
    ap.add_argument("--sigma", type=float, default=-1.0)
    ap.add_argument("--block", type=int, nargs=2, default=[16,16], help="bloque CUDA (bx by)")
    args = ap.parse_args()

    # Leer imagen como RGBA
    img = Image.open(args.input).convert("RGBA")
    W, H = img.size
    src_bytes = img.tobytes()                 # bytes intercalados RGBA
    nbytes_img = W * H * 4

    # Kernel gaussiano 1D (float32) y radio
    k1d, kr = gaussian_kernel_1d(args.mask, args.sigma if args.sigma > 0 else None)
    kbytes = k1d.tobytes()

    # Compilar CUDA
    mod = SourceModule(CUDA_SRC, options=["-O3"])
    k_h = mod.get_function("gauss1d_h_rgba")
    k_v = mod.get_function("gauss1d_v_rgba")

    # Preparar firma explícita (punteros + ints)
    # gauss1d_*_rgba(src*, dst*, int W, int H, k*, int kr)
    k_h.prepare("PPiiPi")
    k_v.prepare("PPiiPi")

    # Reservar en GPU
    d_src = cuda.mem_alloc(nbytes_img)
    d_tmp = cuda.mem_alloc(nbytes_img)
    d_dst = cuda.mem_alloc(nbytes_img)
    d_k   = cuda.mem_alloc(len(kbytes))

    # Subir (no se cronometra)
    cuda.memcpy_htod(d_src, src_bytes)
    cuda.memcpy_htod(d_k, kbytes)

    # Configurar grid/bloque
    bx, by = args.block
    gx = (W + bx - 1) // bx
    gy = (H + by - 1) // by

    # Medir SOLO tiempo GPU (kernels H+V)
    start = cuda.Event(); end = cuda.Event()
    start.record()
    k_h.prepared_call((gx, gy, 1), (bx, by, 1),
                      int(d_src), int(d_tmp),
                      int(W), int(H),
                      int(d_k), int(kr))
    k_v.prepared_call((gx, gy, 1), (bx, by, 1),
                      int(d_tmp), int(d_dst),
                      int(W), int(H),
                      int(d_k), int(kr))
    end.record(); end.synchronize()
    gpu_ms = start.time_till(end)

    # Descargar (no se cronometra)
    out_bytes = bytearray(nbytes_img)
    cuda.memcpy_dtoh(out_bytes, d_dst)

    # Guardar
    out_img = Image.frombytes("RGBA", (W, H), bytes(out_bytes))
    out_path = args.output or (args.input.rsplit(".",1)[0] + f"_gauss_cuda_nonumpy_k"+str(len(k1d))+".png")
    out_img.save(out_path)

    print("\n--- RESUMEN (PyCUDA sin NumPy) ---")
    print(f"Tamaño de imagen: {W}x{H}px")
    print(f"Máscara: {len(k1d)} x {len(k1d)} (separable 1D)")
    print(f"Sigma: {args.sigma if args.sigma > 0 else f'(derivado => {(len(k1d)-1)//2}/3)'}")
    print(f"Bloque: {bx}x{by}  |  Grid: {gx}x{gy}")
    print(f"Tiempo SOLO GPU (H+V): {gpu_ms/1000.0:.6f} s")
    print(f"Salida: {out_path}")