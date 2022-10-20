#include <stdio.h>

const int FACTOR = 2.0;

int scalar_multiply(int x, int y) {
    return x * y;
}

int doubler(int x) {
    return scalar_multiply(x, 2.0);
}

// This is something new

// Test a function which takes two pointers of single float precision

void add(float *x, float *y, float *z, int size) {
    // construct a new pointer dynamically allocated memory
    for (int i = 0; i < size; i++) {
        z[i] = x[i] + y[i];
    }
}