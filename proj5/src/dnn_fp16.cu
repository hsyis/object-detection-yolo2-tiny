#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <time.h>

__global__ void float_to_half(float *in, half *out) {
    int idx = blockIdx.x;
    out[idx] = __float2half(in[idx]);
}

extern "C"
void cuda_mul_float(float *a, float *b, float *c, int m, int k, int n)
{
    clock_t begin;

	cublasHandle_t handle; 
	
	float * d_a;
	float * d_b;
	float * d_c;

	cudaMalloc (( void **)& d_a ,m*k* sizeof (*a)); 
	cudaMalloc (( void **)& d_b ,k*n* sizeof (*b)); 
	cudaMalloc (( void **)& d_c ,m*n* sizeof (*c)); 

    half *d_a_half;
    half *d_b_half;

	cudaMalloc (( void **)& d_a_half ,m*k* sizeof (half));
	cudaMalloc (( void **)& d_b_half ,k*n* sizeof (half));

	cublasCreate (& handle ); 
	
    begin = clock();

	cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); 
	cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); 
	cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); 

    double time_mcpy1 = (double)(clock() - begin) /CLOCKS_PER_SEC;

    // type conv
    begin = clock();

    dim3 gird_a(m * k, 1);
    float_to_half<<<gird_a, 1>>>(d_a, d_a_half);
    dim3 grid_b(k * n, 1);
    float_to_half<<<grid_b, 1>>>(d_b, d_b_half);

    double time_conv = (double)(clock() - begin) /CLOCKS_PER_SEC;

	float al =1.0f; 
	float bet =0.0f; 

    begin = clock();

    // column-major    
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_a_half, CUDA_R_16F, m,
        d_b_half, CUDA_R_16F, k,
        &bet,
        d_c, CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP
    );

    double time_comp = (double)(clock() - begin) /CLOCKS_PER_SEC;

    begin = clock();

	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); 

    double time_mcpy2 = (double)(clock() - begin) /CLOCKS_PER_SEC;

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cudaFree (d_a_half );
	cudaFree (d_b_half );
	cublasDestroy ( handle ); 

    printf("type conversion: %f\n", time_conv);
    printf("memory copy: %f, (up: %f, down: %f)\n", time_mcpy1 + time_mcpy2, time_mcpy1, time_mcpy2);
    printf("computation: %f\n", time_comp);
}

extern "C"
void cuda_max_pool_float(float *a, float *c, int m1, int m2, int m3)
{
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < m3; j++) {
            float max = a[i * m2 * m3 + j];
            for (int k = 1; k < m2; k++) {
                float tmp = a[i * m2 * m3 + k * m3 + j];
                if (tmp > max) {
                    max = tmp;
                }

            }
            c[i * m3 + j] = max;
        }
    }
}

extern "C"
void cuda_norm_float(float *a, float *c, int m, int k, float *gamma, float *mean, float* variance, float epsilon)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = gamma[j] * (a[i * k + j] - mean[j]) / sqrt(variance[j] + epsilon);
        }
    }
}

extern "C"
void cuda_leaky_relu_float(float *a, float *c, int m)
{
    for (int i = 0; i < m; i++) {
        c[i] = max(0.1 * a[i], a[i]);
    }
}

extern "C"
void cuda_max_pool_half(half *a, half *c, int m1, int m2, int m3)
{
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < m3; j++) {
            float max = a[i * m2 * m3 + j];
            for (int k = 1; k < m2; k++) {
                float tmp = a[i * m2 * m3 + k * m3 + j];
                if (tmp > max) {
                    max = tmp;
                }

            }
            c[i * m3 + j] = __float2half(max);
        }
    }
}

extern "C"
void cuda_norm_half(half *a, half *c, int m, int k, float *gamma, float *mean, float* variance, float epsilon)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = __float2half(gamma[j] * (__half2float(a[i * k + j]) - mean[j]) / sqrt(variance[j] + epsilon));
        }
    }
}

extern "C"
void cuda_leaky_relu_half(half *a, half *c, int m)
{
    for (int i = 0; i < m; i++) {
        c[i] = __float2half(max(0.1 * __half2float(a[i]), __half2float(a[i])));
    }
}
