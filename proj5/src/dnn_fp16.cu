#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

extern "C"
void cuda_mul_float(half *a, half *b, half *c, int m, int k, int n)
{
	cublasHandle_t handle; 
	
	half * d_a; 
	half * d_b; 
	half * d_c; 

	cudaMalloc (( void **)& d_a ,m*k* sizeof (*a)); 
	cudaMalloc (( void **)& d_b ,k*n* sizeof (*b)); 
	cudaMalloc (( void **)& d_c ,m*n* sizeof (*c)); 

	cublasCreate (& handle ); 
	
	cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); 
	cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); 
	cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); 

	float al =1.0f; 
	float bet =0.0f; 

    //convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (d_a_16, d_a, m * k);
    //convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (d_b_16, d_b, k * n);

    // column-major    
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_a, CUDA_R_16F, m,
        d_b, CUDA_R_16F, k,
        &bet,
        d_c, CUDA_R_16F, m,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP
    );

	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); 

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cublasDestroy ( handle ); 
}

extern "C"
void cuda_max_pool_float(half *a, half *c, int m1, int m2, int m3)
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
void cuda_norm_float(half *a, half *c, int m, int k, float *gamma, float *mean, float* variance, float epsilon)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i * k + j] = __float2half(gamma[j] * (__half2float(a[i * k + j]) - mean[j]) / sqrt(variance[j] + epsilon));
        }
    }
}

extern "C"
void cuda_leaky_relu_float(half *a, half *c, int m)
{
    for (int i = 0; i < m; i++) {
        c[i] = __float2half(max(0.1 * __half2float(a[i]), __half2float(a[i])));
    }
}
