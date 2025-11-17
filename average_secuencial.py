# promedio_secuencial_puro.py
from PIL import Image

# -------------------------------------------------
# Generar kernel promedio NxN (todo 1/(n*n))
# -------------------------------------------------
def generar_kernel_promedio(n):
    if n % 2 == 0:
        raise ValueError("El tamaño debe ser impar (3, 5, 7...)")
    val = 1.0 / (n * n)
    kernel = [[val for _ in range(n)] for _ in range(n)]
    return kernel

# -------------------------------------------------
# Cargar imagen como matriz 2D (gray[y][x])
# -------------------------------------------------
def cargar_imagen_gris(ruta):
    img = Image.open(ruta).convert("L")
    w, h = img.size
    pix = img.load()

    gray = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            gray[y][x] = pix[x, y]

    return gray, w, h
