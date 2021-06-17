#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C"
void cuda_mul_float(float *a, float *b, float *c, int m, int k, int n)
{
	cublasHandle_t handle; 
	
	float * d_a; 
	float * d_b; 
	float * d_c; 

	cudaMalloc (( void **)& d_a ,m*k* sizeof (*a)); 
	cudaMalloc (( void **)& d_b ,k*n* sizeof (*b)); 
	cudaMalloc (( void **)& d_c ,m*n* sizeof (*c)); 

	cublasCreate (& handle ); 
	
	cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); 
	cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); 
	cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); 

	float al =1.0f; 
	float bet =0.0f; 

    // column-major    
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_a, CUDA_R_32F, m,
        d_b, CUDA_R_32F, k,
        &bet,
        d_c, CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP
    );

	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); 

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cublasDestroy ( handle ); 
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
