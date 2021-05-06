#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>

inline void matrix_kernel16_6(
    const float * __restrict__ A,
	const float * __restrict__ B,
    float * C,
    const int N,
    const int K,
    const int m,
    const int n) 
{
	__m256 mB0;
	__m256 mB1;
	__m256 mA0;
	__m256 mA1;

	__m256 res0_0  = _mm256_set1_ps(0);
	__m256 res1_0  = _mm256_set1_ps(0);
	__m256 res2_0  = _mm256_set1_ps(0);
	__m256 res3_0  = _mm256_set1_ps(0);
	__m256 res4_0  = _mm256_set1_ps(0);
	__m256 res5_0  = _mm256_set1_ps(0);
	__m256 res0_1  = _mm256_set1_ps(0);
	__m256 res1_1  = _mm256_set1_ps(0);
	__m256 res2_1  = _mm256_set1_ps(0);
	__m256 res3_1  = _mm256_set1_ps(0);
	__m256 res4_1  = _mm256_set1_ps(0);
	__m256 res5_1  = _mm256_set1_ps(0);

	for(int k = 0; k < K; k++)
	{
        __builtin_prefetch(&B[N*(k+1)+n+8*0]);
		__builtin_prefetch(&B[N*(k+1)+n+8*1]);

		mB0   = _mm256_loadu_ps(&B[N*k+n+8*0]);
		mB1   = _mm256_loadu_ps(&B[N*k+n+8*1]);

		mA0   = _mm256_set1_ps(A[k+(m+0)*K]);
		mA1   = _mm256_set1_ps(A[k+(m+1)*K]);
		res0_0      = _mm256_fmadd_ps(mB0,mA0,res0_0);
		res0_1      = _mm256_fmadd_ps(mB1,mA0,res0_1);
		res1_0      = _mm256_fmadd_ps(mB0,mA1,res1_0);
		res1_1      = _mm256_fmadd_ps(mB1,mA1,res1_1);

		mA0   = _mm256_set1_ps(A[k+(m+2)*K]);
		mA1   = _mm256_set1_ps(A[k+(m+3)*K]);
		res2_0      = _mm256_fmadd_ps(mB0,mA0,res2_0);
		res2_1      = _mm256_fmadd_ps(mB1,mA0,res2_1);
		res3_0      = _mm256_fmadd_ps(mB0,mA1,res3_0);
		res3_1      = _mm256_fmadd_ps(mB1,mA1,res3_1);

		mA0   = _mm256_set1_ps(A[k+(m+4)*K]);
		mA1   = _mm256_set1_ps(A[k+(m+5)*K]);
		res4_0      = _mm256_fmadd_ps(mB0,mA0,res4_0);
		res4_1      = _mm256_fmadd_ps(mB1,mA0,res4_1);
		res5_0      = _mm256_fmadd_ps(mB0,mA1,res5_0);
		res5_1      = _mm256_fmadd_ps(mB1,mA1,res5_1);
	}

    _mm256_storeu_ps(&C[(m+0)*N+n+0*8], res0_0);
    _mm256_storeu_ps(&C[(m+1)*N+n+0*8], res1_0);
    _mm256_storeu_ps(&C[(m+2)*N+n+0*8], res2_0);
    _mm256_storeu_ps(&C[(m+3)*N+n+0*8], res3_0);
    _mm256_storeu_ps(&C[(m+4)*N+n+0*8], res4_0);
    _mm256_storeu_ps(&C[(m+5)*N+n+0*8], res5_0);
    _mm256_storeu_ps(&C[(m+0)*N+n+1*8], res0_1);
    _mm256_storeu_ps(&C[(m+1)*N+n+1*8], res1_1);
    _mm256_storeu_ps(&C[(m+2)*N+n+1*8], res2_1);
    _mm256_storeu_ps(&C[(m+3)*N+n+1*8], res3_1);
    _mm256_storeu_ps(&C[(m+4)*N+n+1*8], res4_1);
    _mm256_storeu_ps(&C[(m+5)*N+n+1*8], res5_1);
}

struct mtx_arg {
    const float *A;
    const float *B;
    float *C;
    int N;
    int K;
    int m_start;
    int m_end;
};

void *th_func(void *args)
{
    struct mtx_arg *m_arg = (struct mtx_arg *)args;
    const float *A = m_arg->A;
    const float *B = m_arg->B;
    float *C = m_arg->C;
    const int N = m_arg->N;
    const int K = m_arg->K;
    const int m_start = m_arg->m_start;
    const int m_end = m_arg->m_end;

    for (int m = m_start; m < m_end; m += 6) {
        for (int n = 0; n < N; n += 16) {
            matrix_kernel16_6(A, B, C, N, K, m, n);
        }
    }
}

void avx_mul_float(const float *A, const float *B, float *C, int M, int K, int N)
{
    int thread_num = 8;
    pthread_t tid[thread_num];

    int step = (M / (6 * thread_num)) * 6;
    for (int i = 0; i < thread_num; i++) {
        struct mtx_arg *m_arg = (struct mtx_arg *)malloc(sizeof(struct mtx_arg));
        m_arg->A = A;
        m_arg->B = B;
        m_arg->C = C;
        m_arg->N = N;
        m_arg->K = K;

        m_arg->m_start = i * step;
        if (i + 1 != thread_num) {
            m_arg->m_end = (i + 1) * step;
        } else {
            m_arg->m_end = M;
        }

        pthread_create(tid + i, NULL, th_func, (void *)m_arg);
    }

    for (int i = 0; i < thread_num; i++) {
        pthread_join(tid[i], NULL);
    }
}
