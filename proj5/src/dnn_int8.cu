#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <time.h>

__global__ void float_to_int8(float *in, int8_t *out, float max) {
    int idx = blockIdx.x;
    out[idx] = in[idx] * 127 / abs(max);
}

__global__ void postproc(float *in, float max2) {
    int idx = blockIdx.x;
    in[idx] = in[idx] * abs(max2) / (127 * 127);
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

    int8_t * d_a_int8;
    int8_t * d_b_int8;

	cudaMalloc (( void **)& d_a_int8 ,m*k* sizeof (int8_t));
	cudaMalloc (( void **)& d_b_int8 ,k*n* sizeof (int8_t));

	cublasCreate (& handle ); 
	
    begin = clock();

	cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); 
	cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); 
	cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); 

    double time_mcpy1 = (double)(clock() - begin) /CLOCKS_PER_SEC;

    begin = clock();

    int a_max_idx;
    int b_max_idx;

    cublasIsamax(handle, m * k, d_a, 1, &a_max_idx);
    cublasIsamax(handle, k * n, d_b, 1, &b_max_idx);

    float a_max = a[a_max_idx];
    float b_max = b[b_max_idx];

    double time_comp1 = (double)(clock() - begin) /CLOCKS_PER_SEC;

    // type conv
    begin = clock();

    dim3 gird_a(m * k, 1);
    float_to_int8<<<gird_a, 1>>>(d_a, d_a_int8, a_max);
    dim3 grid_b(k * n, 1);
    float_to_int8<<<grid_b, 1>>>(d_b, d_b_int8, b_max);

    double time_conv = (double)(clock() - begin) /CLOCKS_PER_SEC;

	float al =1.0f;
	float bet =0.0f;

    begin = clock();

    // column-major    
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_a_int8, CUDA_R_8I, m,
        d_b_int8, CUDA_R_8I, k,
        &bet,
        d_c, CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP
    );

    dim3 grid_c(m * n, 1);
    postproc<<<grid_c, 1>>>(d_c, a_max * b_max);

    double time_comp2 = (double)(clock() - begin) /CLOCKS_PER_SEC;

    begin = clock();

	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m);

    double time_mcpy2 = (double)(clock() - begin) /CLOCKS_PER_SEC;

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cudaFree (d_a_int8 );
	cudaFree (d_b_int8 );
	cublasDestroy ( handle ); 

    printf("type conversion: %f\n", time_conv);
    printf("memory copy: %f, (up: %f, down: %f)\n", time_mcpy1 + time_mcpy2, time_mcpy1, time_mcpy2);
    printf("computation: %f, (Isamax: %f, GemmEx: %f)\n", time_comp1 + time_comp2, time_comp1, time_comp2);
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
