#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <time.h>

extern "C" {

// global in_node.result for no memory copy
float* d_input;

// input
void init_input(float* input, int size) {
	cudaMalloc(&d_input, size * sizeof(*input));
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
}

void release_all() {
    cudaFree(d_input);
}

__global__ void conv_2d_kernel(
    float* input, float* img_pad, float* x, float* ker,
    int n_in, int h_in, int w_in, int c_in,
    int h_ker, int w_ker, int c_ker, int n_ker,
    int h_stride, int w_stride, int h_out, int w_out, int h_pad, int w_pad)
{
    // pad
    for (int i = 0; i < n_in; i++) {
        for (int j = 0; j < h_in; j++) {
            for (int k = 0; k < w_in; k++) {
                for (int l = 0; l < c_in; l++) {
                    int idx1 = (i * h_in * w_in * c_in) + (j * w_in * c_in) + (k * c_in) + l;
                    int idx2 = (i * (h_in + h_pad) * (w_in + w_pad) * c_in) + ((j + (h_pad / 2)) * (w_in + w_pad) * c_in) + ((k + (w_pad / 2)) * c_in) + l;
                    img_pad[idx2] = input[idx1];
                }
            }
        }
    }

    // im2col
    for (int i = 0; i < h_out; i++) {
        int ih = i * h_stride;
        for (int j = 0; j < w_out; j++) {
            int jw = j * w_stride;
            for (int k = 0; k < n_in; k++) {
                for (int l = 0; l < h_ker; l++) {
                    for (int m = 0; m < w_ker; m++) {
                        for (int n = 0; n < c_in; n++) {
                            x[(k * h_out * w_out * h_ker * w_ker * c_in) + (i * w_out * h_ker * w_ker * c_in) + (j * h_ker * w_ker * c_in) + (l * w_ker * c_in) + (m * c_in) + n]
                                = img_pad[(k * (h_in + h_pad) * (w_in + w_pad) * c_in) + ((ih + l) * (w_in + w_pad) * c_in) + ((jw + m) * c_in) + n];
                        }
                    }
                }
            }
        }
    }
}

// conv_2d
void conv_2d(
    float* ker,
    int n_in, int h_in, int w_in, int c_in,
    int h_ker, int w_ker, int c_ker, int n_ker,
    int h_stride, int w_stride, int h_out, int w_out, int h_pad, int w_pad)
{
    float* d_ker;
    float* d_x;
    float* d_x_t;
    float* d_img_pad;

    int ker_size = h_ker * w_ker * c_ker * n_ker;
    int x_size = n_in * h_out * w_out * h_ker * w_ker * c_in;
    int img_pad_size = n_in * (h_in + h_pad) * (w_in + w_pad) * c_in;

	cudaMalloc(&d_ker, ker_size * sizeof(*d_ker));
    cudaMalloc(&d_x, x_size * sizeof(*d_x));
    cudaMalloc(&d_x_t, x_size * sizeof(*d_x_t));
    cudaMalloc(&d_img_pad, img_pad_size * sizeof(*d_img_pad));

    cudaMemcpy(d_ker, ker, ker_size, cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, x_size * sizeof(*d_x));
    cudaMemset(d_x_t, 0, x_size * sizeof(*d_x_t));
    cudaMemset(d_img_pad, 0, img_pad_size * sizeof(*d_img_pad));

    conv_2d_kernel<<<1, 1>>>(d_input, d_img_pad, d_x, d_ker, n_in, h_in, w_in, c_in, h_ker, w_ker, c_ker, n_ker, h_stride, w_stride, h_out, w_out, h_pad, w_pad);

    // TODO: row-major to column-major

    // mat mul
    int m = n_in * h_out * w_out;
    int k = h_ker * w_ker * c_in;
    int n = n_ker;

    cudaFree(d_input);
	cudaMalloc(&d_input, m * n * sizeof(*d_input));

	cublasHandle_t handle; 
	cublasCreate(&handle); 

	float al =1.0f; 
	float bet =0.0f; 

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &al,
        d_x_t, CUDA_R_32F, m,
        d_ker, CUDA_R_32F, k,
        &bet,
        d_input, CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP
    );

    cudaFree(d_ker);
    cudaFree(d_x);
    cudaFree(d_x_t);
    cudaFree(d_img_pad);
    cudaFree(d_input);
	cublasDestroy (handle); 
}

} // extern "C"

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

	cublasCreate (& handle ); 
	
    begin = clock();

	cublasSetMatrix (m,k, sizeof (*a) ,a,m,d_a ,m); 
	cublasSetMatrix (k,n, sizeof (*b) ,b,k,d_b ,k); 
	cublasSetMatrix (m,n, sizeof (*c) ,c,m,d_c ,m); 

    double time_mcpy1 = (double)(clock() - begin) /CLOCKS_PER_SEC;

	float al =1.0f; 
	float bet =0.0f; 

    begin = clock();

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

    double time_comp = (double)(clock() - begin) /CLOCKS_PER_SEC;

    begin = clock();

	cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); 

    double time_mcpy2 = (double)(clock() - begin) /CLOCKS_PER_SEC;

	cudaFree (d_a ); 
	cudaFree (d_b ); 
	cudaFree (d_c ); 
	cublasDestroy ( handle ); 

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
