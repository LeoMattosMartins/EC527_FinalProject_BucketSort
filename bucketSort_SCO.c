#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void bucketSort_SCO(float *arr, int N, int K, float A, float B) {
    /* (1) count per bucket */
    int *counts = calloc(K, sizeof(int));
    float invRange = K / (B - A);

    for (int ii = 0; ii < N; ii++) {
        int idx = (int)((arr[ii] - A) * invRange);
        counts[idx]++;
    }

    /* (2) prefix-sum offsets */
    int *offsets = malloc(K * sizeof(int));
    offsets[0] = 0;
    for (int bb = 1; bb < K; bb++) {
        offsets[bb] = offsets[bb - 1] + counts[bb - 1];
    }

    /* (3) allocate aligned storage for N floats */
    float *storage = _mm_malloc(N * sizeof(float), 32);

    /* (4) prepare write pointers */
    int *wp = malloc(K * sizeof(int));
    memcpy(wp, offsets, K * sizeof(int));

    /* (5) SIMD distribution: 8 floats at a time */
    __m256 a_vec   = _mm256_set1_ps(A);
    __m256 inv_vec = _mm256_set1_ps(invRange);

    int ii = 0, limit = N - 7;
    for (; ii <= limit; ii += 8) {
        __m256 v = _mm256_loadu_ps(&arr[ii]);
        __m256 tmp = _mm256_sub_ps(v, a_vec);
        tmp = _mm256_mul_ps(tmp, inv_vec);

        alignas(32) int idxs[8];
        _mm256_store_si256((__m256i*)idxs, _mm256_cvttps_epi32(tmp));

        float vals[8];
        _mm256_storeu_ps(vals, v);

        for (int jj = 0; jj < 8; ++jj) {
            int id = idxs[jj];
            storage[ wp[id]++ ] = vals[jj];
        }
    }

    for (; ii < N; ii++) {
        int id = (int)((arr[ii] - A) * invRange);
        storage[wp[id]++] = arr[ii];
    }

    /* (6) per-bucket radix sort + SIMD merge-back */
    uint32_t *bits    = (uint32_t*) storage;
    uint32_t *tmp_buf = malloc(N * sizeof(uint32_t));
    int dest = 0;

    for (int bb = 0; bb < K; bb++) {
        int start = offsets[bb];
        int cnt   = wp[bb] - start;
        uint32_t *base = bits + start;

        if (cnt > 0) {
            for (int pp = 0; pp < 4; pp++) {
                int shift = pp << 3;
                int hist[256] = {0};

                /* histogram */
                for (int jj = 0; jj < cnt; jj++) {
                    hist[(base[jj] >> shift) & 0xFF]++;
                }

                /* prefix-sum */
                int sum = 0;
                for (int mm = 0; mm < 256; mm++) {
                    int t = hist[mm];
                    hist[mm] = sum;
                    sum += t;
                }

                /* scatter */
                for (int jj = 0; jj < cnt; jj++) {
                    int key = (base[jj] >> shift) & 0xFF;
                    tmp_buf[ hist[key]++ ] = base[jj];
                }

                /* swap buffers */
                uint32_t *swap = base;
                base = tmp_buf;
                tmp_buf = swap;
            }
        }

        /* SIMD merge-back: 8 floats at a time */
        int full8 = cnt >> 3;
        for (int jj = 0; jj < full8; jj++) {
            alignas(32) uint32_t u[8];
	    int jj8 = jj * 8;
            for (int tt = 0; tt < 8; tt++) {
                uint32_t w = base[jj8 + tt];
                w = (w - 0x80000000u) ^ ((int32_t)w >> 31);
                u[tt] = w;
            }
            __m256 out = _mm256_castsi256_ps(_mm256_load_si256((__m256i*)u));
            _mm256_storeu_ps(&arr[dest + jj8, out);
        }

        /* tail */
        for (int jj = (full8 << 3); jj < cnt; jj++) {
            uint32_t w = base[jj];
            w = (w - 0x80000000u) ^ ((int32_t)w >> 31);
            arr[dest + jj] = *(float*)&w;
        }

        dest += cnt;
    }

    /* (7) cleanup */
    free(counts);
    free(offsets);
    free(wp);
    _mm_free(storage);
    free(tmp_buf);
}
