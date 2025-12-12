#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

static inline __m256 _mm256_abs_ps(__m256 x) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(x, mask);
}

void sqrtAVX2(int N,
                float initialGuess,
                float values[],
                float output[])
{
    // Match the serial/ISPC algorithm: iterate on guess for 1/sqrt(x)
    // until |guess^2 * x - 1| <= kThreshold.
    const __m256 kThreshold = _mm256_set1_ps(0.00001f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 threeHalves = _mm256_set1_ps(1.5f);

    int i = 0;
    const int width = 8;
    const int limit = N - (N % width);

    for (; i < limit; i += width) {
        __m256 x = _mm256_loadu_ps(values + i);
        __m256 guess = _mm256_set1_ps(initialGuess);

        __m256 g2 = _mm256_mul_ps(guess, guess);
        __m256 error = _mm256_abs_ps(_mm256_sub_ps(_mm256_mul_ps(g2, x), one));

        __m256 active = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ);
        while (_mm256_movemask_ps(active) != 0) {
            // Same update as serial:
            //   guess = (3*guess - x*guess^3) * 0.5
            // Algebraically equivalent (fewer multiplies):
            //   guess = guess * (1.5 - 0.5 * x * guess^2)
            g2 = _mm256_mul_ps(guess, guess);
            __m256 step = _mm256_sub_ps(threeHalves, _mm256_mul_ps(_mm256_mul_ps(half, x), g2));
            __m256 nextGuess = _mm256_mul_ps(guess, step);

            // Per-lane control flow: only update lanes still above threshold.
            guess = _mm256_blendv_ps(guess, nextGuess, active);

            g2 = _mm256_mul_ps(guess, guess);
            error = _mm256_abs_ps(_mm256_sub_ps(_mm256_mul_ps(g2, x), one));
            active = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ);
        }

        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, guess));
    }

    // Tail (scalar, identical to sqrtSerial)
    static const float kThresholdScalar = 0.00001f;
    for (; i < N; i++) {
        float x = values[i];
        float guess = initialGuess;
        float error = fabsf(guess * guess * x - 1.f);
        while (error > kThresholdScalar) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabsf(guess * guess * x - 1.f);
        }
        output[i] = x * guess;
    }
}