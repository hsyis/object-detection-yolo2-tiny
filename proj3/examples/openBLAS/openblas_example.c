#include <stdio.h>
#include <cblas.h>

#define M 2
#define N 2
#define K 3
#define X 4
#define Y 5

void print_mat(const char *name, int r, int c, float *m)
{
    printf("%s =\n", name);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2lf ", m[i * c + j]);
        }
        printf("\n");
    }
}

int main()
{
    float A[M * K] = {
        1, 2, 3,
        4, 5, 6,
    };
    float B[K * N] = {
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
    };
    float C[X * Y] = { 0 };

    // M x K, K x N -> M x N (single precision, 's'gemm)
    // Save A x B in C starting from (1, 1)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1,
        A, K,
        B, N,
        0,
        C + Y + 1, Y
    );

    print_mat("A", M, K, A);
    print_mat("B", K, N, B);
    print_mat("C", X, Y, C);

    return 0;
}
