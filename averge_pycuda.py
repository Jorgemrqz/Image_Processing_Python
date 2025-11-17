import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
from time import perf_counter

# -------------------------------------------------
# CUDA: filtro promedio dinámico NxN
# -------------------------------------------------
kernel_code = r"""
__global__ void mean_filter(
    unsigned char *img,
    unsigned char *out,
    int width,
    int height,
    int n
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pad = n / 2;
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int xx = x + j - pad;
            int yy = y + i - pad;

            // Padding por borde (clamp)
            if (xx < 0) xx = 0;
            if (yy < 0) yy = 0;
            if (xx >= width)  xx = width  - 1;
            if (yy >= height) yy = height - 1;

            unsigned char pixel = img[yy * width + xx];
            sum += (float)pixel;
        }
    }

    float denom = (float)(n * n);
    float val = sum / denom;  // promedio

    if (val < 0.0f)   val = 0.0f;
    if (val > 255.0f) val = 255.0f;

    // redondeo
    out[y * width + x] = (unsigned char)(val + 0.5f);
}
"""

mod = SourceModule(kernel_code)
mean_filter_gpu = mod.get_function("mean_filter")


# -------------------------------------------------
# Programa principal
# -------------------------------------------------
if __name__ == "__main__":
    print("=== Filtro Promedio Dinámico con PyCUDA ===")

    # Tamaño de kernel
    try:
        n = int(input("Ingrese tamaño del kernel (impar): ").strip())
    except ValueError:
        n = 3

    if n % 2 == 0:
        print("El tamaño debe ser impar, usando 3x3.")
        n = 3

    # Cargar imagen (debe existir original.jpg)
    img = Image.open("original.jpg").convert("L")
    gray = np.array(img).astype(np.uint8)
    H, W = gray.shape

    # Salida
    out = np.zeros_like(gray, dtype=np.uint8)

    # Reservar en GPU
    d_img = drv.mem_alloc(gray.nbytes)
    d_out = drv.mem_alloc(out.nbytes)

    # Copiar imagen de host a device (fuera del tiempo medido)
    drv.memcpy_htod(d_img, gray)

    # Configurar grid y block
    block = (16, 16, 1)
    grid = ((W + block[0] - 1) // block[0],
            (H + block[1] - 1) // block[1],
            1)

    # ----------------- medir solo cómputo GPU -----------------
    t0 = perf_counter()
    mean_filter_gpu(
        d_img,
        d_out,
        np.int32(W),
        np.int32(H),
        np.int32(n),
        block=block,
        grid=grid
    )
    drv.Context.synchronize()  # asegurar fin del kernel
    t1 = perf_counter()
    # -----------------------------------------------------------

    print(f"\nTiempo GPU filtro promedio {n}x{n}: {(t1 - t0)*1000:.2f} ms")

    # Copiar resultado a host y guardar
    drv.memcpy_dtoh(out, d_out)
    out_img = Image.fromarray(out)
    out_name = f"promedio_gpu_{n}x{n}.jpg"
    out_img.save(out_name)
    print("Imagen guardada como:", out_name)