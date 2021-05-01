/*
 * Reference:
 * https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
 */

#include <immintrin.h>
#include <stdio.h>

int main()
{
    // Multiply 8 floats at a time
    __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    __m256 odds  = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
    __m256 res   = _mm256_mul_ps(evens, odds);

    printf("evens: ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&evens[i]);
    printf("\nodds:  ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&odds[i]);
    printf("\nres:   ");
    for (int i = 0; i < 8; i++) printf("%3.0f ", *(float *)&res[i]);
    printf("\n");

    return 0;
}
