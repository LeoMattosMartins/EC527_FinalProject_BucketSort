#include <stdlib.h>

void bucketSort_baseline(float *arr, int N, int K, float A, float B) {
    /* (1)  Pre-allocate: each bucket can hold up to N elements */
    float **buckets = (float**) malloc(K * sizeof(float*));
    int *counts = (int*) malloc(K * sizeof(int));

    for (int ii = 0; ii < K; ii++) {
        buckets[ii] = (float*) malloc(N * sizeof(float));
        counts[ii] = 0;
    }

    /* (2) Distribute into buckets */
    for (int ii = 0; ii < N; ii++) {
        /* (2a) calculate index */ 
	int idx = (int)(((arr[ii] - A) / (B - A)) * K);

	/* (2b) place element in corresponding bucket	
        buckets[idx][counts[idx]++] = arr[i];
    }

    /* (3) Sort each bucket (insertion sort) and write back */
    int pos = 0;
    for (int bb = 0; bb < K; bb++) {
        for (int ii = 1; ii < counts[bb]; ii++) {
            float key = buckets[bb][ii];
            int jj = ii - 1;
            while (jj >= 0 && buckets[bb][jj] > key) {
                buckets[bb][jj + 1] = buckets[bb][jj];
                jj--;
            }
            buckets[bb][jj + 1] = key;
        }
        /* (4) Merge */
        for (int ii = 0; ii < counts[bb]; ii++) {
            arr[pos++] = buckets[bb][ii];
        }
        free(buckets[bb]);
    }

    free(buckets);
    free(counts);
}

