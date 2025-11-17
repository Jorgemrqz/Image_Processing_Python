# sobel_pycuda.py

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
from time import perf_counter

print("=== Sobel PyCUDA (GPU) ===")

# -----------------------------
# Generar máscaras Sobel dinámicas
# -----------------------------
def generar_mascara_sobel(n):
    c = n // 2
    Kx = np.zeros((n, n), dtype=np.float32)
    Ky = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            Kx[i, j] = (j - c) * (abs(i - c) + 1)
            Ky[i, j] = (i - c) * (abs(j - c) + 1)

    return Kx, Ky


def main():
    print("Máscara Sobel lista.")

if __name__ == "__main__":
    main()
