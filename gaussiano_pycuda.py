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
