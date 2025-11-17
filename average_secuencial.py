# -------------------------------------------------
# Generar kernel promedio NxN (todo 1/(n*n))
# -------------------------------------------------
def generar_kernel_promedio(n):
    if n % 2 == 0:
        raise ValueError("El tamaño debe ser impar (3, 5, 7...)")
    val = 1.0 / (n * n)
    kernel = [[val for _ in range(n)] for _ in range(n)]
    return kernel
