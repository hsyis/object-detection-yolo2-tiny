# include <cuda_runtime.h>
# include "cublas_v2.h"

extern "C"
void cublas_mul_float(float *a, float *b, float *c, int m, int k, int n)
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
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_a, m,
        d_b, k,
        &bet,
        d_c, m
    );
	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); 

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cublasDestroy ( handle ); 
}
