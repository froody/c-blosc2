#include <stdio.h>
#include <string.h>
#include "blosc2.h"
#include "cat_pack.h"

#define BSIZE 1024

int test_packing(uint8_t bits_per_val, uint8_t *data, int num_classes) {
    uint8_t dest[BSIZE];
    uint8_t decompressed[BSIZE];
    
    int32_t rc = cat_pack_forward(data, dest, BSIZE, bits_per_val, NULL, 0);
    if (rc != BLOSC2_ERROR_SUCCESS) {
        printf("Forward failed for bits_per_val=%d\n", bits_per_val);
        return -1;
    }
    
    rc = cat_pack_backward(dest, decompressed, BSIZE, bits_per_val, NULL, 0);
    if (rc != BLOSC2_ERROR_SUCCESS) {
        printf("Backward failed for bits_per_val=%d\n", bits_per_val);
        return -1;
    }
    
    if (memcmp(data, decompressed, BSIZE) != 0) {
        printf("Mismatch for bits_per_val=%d\n", bits_per_val);
        for(int i=0; i<32; i++) {
            printf("Expected %d, got %d\n", data[i], decompressed[i]);
        }
        return -1;
    }
    
    printf("Passed bits_per_val=%d with %d classes\n", bits_per_val, num_classes);
    return 0;
}

int main() {
    uint8_t data[BSIZE];
    uint8_t dest[BSIZE];
    
    // 1-bit test
    for (int i = 0; i < BSIZE; i++) data[i] = (i % 2) ? 100 : 200;
    if (test_packing(1, data, 2) < 0) return -1;
    
    // 2-bit test
    uint8_t classes2[4] = {10, 20, 30, 40};
    for (int i = 0; i < BSIZE; i++) data[i] = classes2[i % 4];
    if (test_packing(2, data, 4) < 0) return -1;
    
    // 4-bit test
    uint8_t classes4[16];
    for (int i = 0; i < 16; i++) classes4[i] = i * 10;
    for (int i = 0; i < BSIZE; i++) data[i] = classes4[i % 16];
    if (test_packing(4, data, 16) < 0) return -1;
    
    // 8-bit test (noop)
    for (int i = 0; i < BSIZE; i++) data[i] = i % 256;
    if (test_packing(8, data, 256) < 0) return -1;
    
    // Negative test: 1-bit requested but data has 3 classes
    for (int i = 0; i < BSIZE; i++) data[i] = i % 3;
    if (cat_pack_forward(data, dest, BSIZE, 1, NULL, 0) == BLOSC2_ERROR_FAILURE) {
        printf("Passed negative test: 1-bit with 3 classes failed correctly\n");
    } else {
        printf("Failed negative test: 1-bit with 3 classes should fail\n");
        return -1;
    }
    
    printf("All tests passed.\n");
    return 0;
}
