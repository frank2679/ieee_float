#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    float a[1] = {1.0e-45f};
    float b[1] = {1.0e-45f};
    float c[1] = {0.0f};

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(float));
    cudaMalloc((void **)&d_b, sizeof(float));
    cudaMalloc((void **)&d_c, sizeof(float));

    cudaMemcpy(d_a, a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float), cudaMemcpyHostToDevice);

    add<<<1, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    printf("a = %e\n", a[0]);
    printf("b = %e\n", b[0]);
    printf("sum = %e\n", c[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
