#include "supa.h"
#include <math.h>
#include <stdio.h>
__global__ void add(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(void) {

    suError_t err = suSuccess;
    int numElements = 1;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = (float){1.0e-45f};
        h_B[i] = (float){1.0e-45f};
    }

    float *d_A = NULL;
    err = suMallocDevice((void **)&d_A, size);
    if (err != suSuccess) {
        fprintf(stderr, "Allocate device vector A error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = suMallocDevice((void **)&d_B, size);
    if (err != suSuccess) {
        fprintf(stderr, "Allocate device vector B error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = suMallocDevice((void **)&d_C, size);
    if (err != suSuccess) {
        fprintf(stderr, "Allocate device vector C error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }

    err = suMemcpy(d_A, h_A, size);
    if (err != suSuccess) {
        fprintf(stderr, "H2D Memory Copy A error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }
    err = suMemcpy(d_B, h_B, size);
    if (err != suSuccess) {
        fprintf(stderr, "H2D Memory Copy B error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }

    // one CU run two batch, 32x4x2 = 256
    dim3 blockDim(1);
    dim3 gridDim(1);

    suLaunchKernel(add, gridDim, blockDim, 0, NULL, d_A, d_B, d_C);
    suDeviceSynchronize();
    assert(suGetLastError() == suSuccess);
    err = suMemcpy(h_C, d_C, size);
    if (err != suSuccess) {
        fprintf(stderr, "D2H Memory Copy C error (%d)!\n", err);
        exit(EXIT_FAILURE);
    }

    // Final check
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          return EXIT_FAILURE;
        }
    }

    printf("a = %e\n", h_A[0]);
    printf("b = %e\n", h_B[0]);
    printf("sum = %e\n", h_C[0]);

    // Free device and host memory
    suFree(d_A);
    suFree(d_B);
    suFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return EXIT_SUCCESS;
}
