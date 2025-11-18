# promedio_secuencial_puro.py
from PIL import Image
from time import perf_counter
import warnings

warnings.simplefilter("ignore", Image.DecompressionBombWarning)

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
# -------------------------------------------------
# Filtro promedio Python puro
# -------------------------------------------------
def promedio_puro(gray, kernel):
    n = len(kernel)
    pad = n // 2

    h = len(gray)
    w = len(gray[0])

    # Crear imagen con padding
    ph = h + 2 * pad
    pw = w + 2 * pad
    padded = [[0.0 for _ in range(pw)] for _ in range(ph)]

    # Copiar con borde repetido
    for y in range(ph):
        for x in range(pw):
            yy = min(max(y - pad, 0), h - 1)
            xx = min(max(x - pad, 0), w - 1)
            padded[y][x] = float(gray[yy][xx])

    # Aplicar convolución
    salida = Image.new("L", (w, h))
    out = salida.load()

    for y in range(h):
        for x in range(w):
            suma = 0.0
            for i in range(n):
                for j in range(n):
                    suma += padded[y + i][x + j] * kernel[i][j]

            # Clamp
            val = int(suma)
            if val < 0: val = 0
            if val > 255: val = 255
            out[x, y] = val

    return salida

# -------------------------------------------------
# Programa principal
# -------------------------------------------------
if __name__ == "__main__":
    print("=== Filtro Promedio Secuencial (Python puro) ===")

    try:
        n = int(input("Ingrese el tamaño del kernel (impar): "))
    except:
        n = 3

    if n % 2 == 0:
        print("Kernel par detectado, usando 3x3.")
        n = 3

    # Cargar imagen
    gray, w, h = cargar_imagen_gris("original.jpg")

    # Crear kernel promedio
    kernel = generar_kernel_promedio(n)

    # Ejecutar filtro
    t0 = perf_counter()
    out_img = promedio_puro(gray, kernel)
    t1 = perf_counter()

    print(f"\nTiempo Filtro Promedio {n}x{n}: {(t1 - t0)*1000:.2f} ms")

    # Guardar imagen
    out_name = f"promedio_{n}x{n}.jpg"
    out_img.save(out_name)
    print("Imagen guardada como:", out_name)

