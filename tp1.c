#include "tp1.h"
#include <stdlib.h>
#include <math.h>

bool is_prime(int x){
    if (x == 2){
        return true;
    }
    
    if (x % 2 == 0){
        return false;
    }

    if (x % 3 == 0){
        return false;
    }

    for (int i = 5; i <= sqrt(x); i += 6){
        if (x % i == 0 || x % (i + 2) == 0){
            return false;
        }
    }

    return true;
}

//asjdjkasjdjkasjkdjaskdjaskjdkdjkajaksdjaskd

int storage_capacity(float d, float v){
    return 0;
}

void swap(int *x, int *y) {
    return;
}

int array_max(const int *array, int length) {
    return 0;
}

void array_map(int *array, int length, int f(int)) {
    return;
}

int *copy_array(const int *array, int length) {
    return NULL;
}

int **copy_array_of_arrays(const int **array_of_arrays, const int *array_lenghts, int array_amount){
    return NULL;
}

void free_array_of_arrays(int **array_of_arrays, int *array_lenghts, int array_amount){
    return;
}

void bubble_sort(int *array, int length){
    return;
}

bool array_equal(const int *array1, int length1, const int *array2, int length2){
    return true;
}

bool integer_anagrams(const int *array1, int length1,
                      const int *array2, int length2){
    return true;
}