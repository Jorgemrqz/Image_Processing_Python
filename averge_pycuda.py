from pycuda.compiler import SourceModule

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

            if (xx < 0) xx = 0;
            if (yy < 0) yy = 0;
            if (xx >= width)  xx = width  - 1;
            if (yy >= height) yy = height - 1;

            unsigned char pixel = img[yy * width + xx];
            sum += (float)pixel;
        }
    }

    float denom = (float)(n * n);
    float val = sum / denom;

    if (val < 0.0f)   val = 0.0f;
    if (val > 255.0f) val = 255.0f;

    out[y * width + x] = (unsigned char)(val + 0.5f);
}
"""

mod = SourceModule(kernel_code)
mean_filter_gpu = mod.get_function("mean_filter")
