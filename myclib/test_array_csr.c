#include <stdlib.h>
#include <stdio.h>


struct CsrMatrix {
    float *x;
    int size;
    int a;
};

void my_c_struct_ptr(struct CsrMatrix x[], size_t len) {
    printf("problem size %i\n", len);
    for (int i = 0; i < len; i++) {
        printf("array size: %i\n", x[i].size);
        printf("action %i\n", x[i].a);
        printf("array value: %f\n", x[i].x[0]);
    }
}