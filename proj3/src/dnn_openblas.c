#include <cblas.h>

void openblas_mul_double(double *a, double *b, double *c, int m, int k, int n)
{
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        1.,
        a, k,
        b, n,
        0.,
        c, n
    );
}
