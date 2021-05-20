#include <stdio.h>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

__device__ float get_element(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void set_element(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

 __device__ Matrix get_sub_matrix(Matrix A, int row, int col) 
{
    Matrix a_sub;
    a_sub.width    = BLOCK_SIZE;
    a_sub.height   = BLOCK_SIZE;
    a_sub.stride   = A.stride;
    a_sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return a_sub;
}

 __global__ void mat_mul_kernel(Matrix A, Matrix B, Matrix C)
{
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    Matrix c_sub = get_sub_matrix(C, block_row, block_col);

    float c_value = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix a_sub = get_sub_matrix(A, block_row, m);
        Matrix b_sub = get_sub_matrix(B, m, block_col);

        __shared__ float a_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float b_s[BLOCK_SIZE][BLOCK_SIZE];

        a_s[row][col] = get_element(a_sub, row, col);
        b_s[row][col] = get_element(b_sub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            c_value += a_s[row][e] * b_s[e][col];

        __syncthreads();
    }

    set_element(c_sub, row, col, c_value);
}

extern "C"
void cuda_mul_float(float *a, float *b, float *c, int m, int k, int n)
{
    int m16 = m % 16 ? m + (16 - m % 16) : m;
    int k16 = k % 16 ? k + (16 - k % 16) : k;
    int n16 = n % 16 ? n + (16 - n % 16) : n;

    float *a16_16 = (float *)calloc(m16 * k16, sizeof(float *));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a16_16[i * k16 + j] = a[i * k + j];
        }
    }

    float *b16_16 = (float *)calloc(k16 * n16, sizeof(float *));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b16_16[i * n16 + j] = b[i * n + j];
        }
    }

    float *c16_16 = (float *)calloc(m16 * n16, sizeof(float *));

    Matrix A = { .width = k16, .height = m16, .stride = k16, .elements = a16_16 };
    Matrix B = { .width = n16, .height = k16, .stride = n16, .elements = b16_16 };
    Matrix C = { .width = n16, .height = m16, .stride = n16, .elements = c16_16 };

    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;

    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;

    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;

    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    mat_mul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = c16_16[i * n16 + j];
        }
    }

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    free(a16_16);
    free(b16_16);
    free(c16_16);
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
