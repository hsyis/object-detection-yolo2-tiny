/*
 * Reference:
 * https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
 */

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 32
#define n 32
#define k 32

#include <stddef.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/time.h>

static double second(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

__global__ void int2float(float *dout, int8_t *din) {
    int i = threadIdx.x;
    dout[i] = (float)(din[i]);
}

__global__ void float2int(int8_t *dout, float *din) {
    int i = threadIdx.x;
    dout[i] = (int8_t)(din[i]);
}

int main(int argc, char *argv[]) {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float *a;
  float *b;
  float *c;
  double start, end;
  cublasMath_t mode;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <ON|OFF>\n", argv[0]);
    fprintf(stderr, "\tSpecify whether you want to turn on the Tensor Cores\n");
    exit(1);
  }
  if (!strcmp(argv[1], "ON")) {
    printf("Enabling Tensor Core\n");
    mode = CUBLAS_TENSOR_OP_MATH;
  } else if (!strcmp(argv[1], "OFF")) {
    printf("Disabling Tensor Core\n");
    mode = CUBLAS_DEFAULT_MATH;
  } else {
    fprintf(stderr, "Invalid argument\n");
    exit(1);
  }

  a = (float *)malloc(m * k * sizeof(float));
  b = (float *)malloc(k * n * sizeof(float));
  c = (float *)malloc(m * n * sizeof(float));

  float ind = 1;
  for (j = 0; j < k; j++) {
    for (i = 0; i < m; i++) {
      a[IDX2C(i, j, m)] = ind;
    }
  }

  ind = 1;
  for (j = 0; j < n; j++) {
    for (i = 0; i < k; i++) {
      b[IDX2C(i, j, k)] = ind;
    }
  }

  float *d_a;
  int8_t *d_a_i8;
  float *d_b;
  int8_t *d_b_i8;
  float *d_c;

  cudaStat = cudaMalloc((void **)&d_a, m * k * sizeof(*a));
  cudaStat = cudaMalloc((void **)&d_b, k * n * sizeof(*b));
  cudaStat = cudaMalloc((void **)&d_c, m * n * sizeof(*c));

  cudaStat = cudaMalloc((void **)&d_a_i8, m * k * sizeof(int8_t));
  cudaStat = cudaMalloc((void **)&d_b_i8, k * n * sizeof(int8_t));

  stat = cublasCreate(&handle);

  cublasSetMathMode(handle, mode);

  stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m);
  stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k);
  stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m);

  
  float2int<<<1, m*k>>>(d_a_i8, d_a);
  float2int<<<1, k*n>>>(d_b_i8, d_b);

  float al = 1.0f;
  float bet = 0.0f;

  start = second();
  for (int i = 0; i < 10000; i++) {
    if ((stat = cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k, 
                    &al, 
                    d_a_i8, CUDA_R_8I, m,
                    d_b_i8, CUDA_R_8I, k,
                    &bet, 
                    d_c, CUDA_R_32F, m,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                    )
                ) != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "GemmEx failed\n");
      exit(1);
    }
  }
  end = second();

  if ((stat != cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m)) != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cublasGetMatrix failed\n");
      exit(1);
  }

  printf("c after Sgemm :\n");
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf(" %.2f", c[IDX2C(i, j, m)]);
    }
    printf("\n");
  }
  printf("Time took: %lf\n", end - start);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  free(a);
  free(b);
  free(c);
  return EXIT_SUCCESS;
}
