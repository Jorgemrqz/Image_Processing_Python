# sobel_pycuda.py

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
from time import perf_counter

print("=== Sobel PyCUDA (GPU) ===")


def generar_mascara_sobel(n):
    c = n // 2
    Kx = np.zeros((n, n), dtype=np.float32)
    Ky = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            Kx[i, j] = (j - c) * (abs(i - c) + 1)
            Ky[i, j] = (i - c) * (abs(j - c) + 1)
    return Kx, Ky


kernel_code = """
__global__ void sobel_mag(
    unsigned char *img,
    float *mag,
    float *Kx, float *Ky,
    int width, int height,
    int n
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pad = n / 2;
    float gx = 0.0f;
    float gy = 0.0f;

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            int xx = x + j - pad;
            int yy = y + i - pad;

            if (xx < 0) xx = 0;
            if (yy < 0) yy = 0;
            if (xx >= width) xx = width - 1;
            if (yy >= height) yy = height - 1;

            unsigned char pixel = img[yy * width + xx];

            gx += pixel * Kx[i*n + j];
            gy += pixel * Ky[i*n + j];
        }
    }

    mag[y * width + x] = sqrtf(gx * gx + gy * gy);
}

__global__ void normalize_mag(
    float *mag,
    unsigned char *out,
    float max_val,
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float v = mag[y*width + x];
    v = (v / max_val) * 255.0f;

    if (v < 0) v = 0;
    if (v > 255) v = 255;

    out[y*width + x] = (unsigned char)v;
}
"""

mod = SourceModule(kernel_code)
sobel_mag_gpu = mod.get_function("sobel_mag")
normalize_gpu = mod.get_function("normalize_mag")


def main():
    print("Kernels compilados.")

if __name__ == "__main__":
    main()
