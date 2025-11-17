# sobel_pycuda.py

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
from time import perf_counter

print("=== Sobel PyCUDA (GPU) ===")


# ------------------------------------
# Generar máscara Sobel dinámica
# ------------------------------------
def generar_mascara_sobel(n):
    c = n // 2
    Kx = np.zeros((n, n), dtype=np.float32)
    Ky = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            Kx[i, j] = (j - c) * (abs(i - c) + 1)
            Ky[i, j] = (i - c) * (abs(j - c) + 1)
    return Kx, Ky


# ------------------------------------
# Kernels CUDA Sobel
# ------------------------------------
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
    print("=== Sobel Dinámico PyCUDA (Normalización completa) ===")

    n = int(input("Tamaño de máscara (impar): "))
    if n % 2 == 0:
        n = 3
        print("Usando 3x3.")

    img = Image.open("original.jpg").convert("L")
    gray = np.array(img).astype(np.uint8)
    H, W = gray.shape

    # Generar máscaras
    Kx, Ky = generar_mascara_sobel(n)
    Kx_flat = Kx.reshape(-1).astype(np.float32)
    Ky_flat = Ky.reshape(-1).astype(np.float32)

    # Reservar memoria GPU
    mag_gpu = drv.mem_alloc(gray.size * 4)   # float32
    out_gpu = drv.mem_alloc(gray.size)       # uint8

    # Configuración CUDA
    block = (16, 16, 1)
    grid = ((W + 15) // 16, (H + 15) // 16, 1)

    # -------------------------------
    # KERNEL 1: calcular magnitud
    # -------------------------------
    t0 = perf_counter()
    sobel_mag_gpu(
        drv.In(gray),
        mag_gpu,
        drv.In(Kx_flat),
        drv.In(Ky_flat),
        np.int32(W),
        np.int32(H),
        np.int32(n),
        block=block, grid=grid
    )
    t1 = perf_counter()

    print(f"Tiempo magnitud GPU: {(t1 - t0)*1000:.2f} ms")

    # Descargar magnitud para obtener max_val
    mag_host = np.empty_like(gray, dtype=np.float32)
    drv.memcpy_dtoh(mag_host, mag_gpu)

    max_val = float(mag_host.max())
    if max_val == 0:
        max_val = 1.0

    # -------------------------------
    # KERNEL 2: normalizar 0–255
    # -------------------------------
    normalize_gpu(
        drv.In(mag_host),
        out_gpu,
        np.float32(max_val),
        np.int32(W),
        np.int32(H),
        block=block, grid=grid
    )

    # Obtener imagen final normalizada
    out_host = np.empty_like(gray)
    drv.memcpy_dtoh(out_host, out_gpu)

    out_img = Image.fromarray(out_host)
    filename = f"sobel_gpu_{n}x{n}.jpg"
    out_img.save(filename)

    print("Imagen guardada:", filename)


if __name__ == "__main__":
    main()
