#include "tp1.h"
#include <stdlib.h>
#include <math.h>

bool is_prime(int x){
    if (x <= 1) return false;

    if (x == 2){
        return true;
    }
    
    if (x % 2 == 0 || x % 3 == 0){
        return false;
    }

    for (int i = 5; i <= sqrt(x); i += 6){
        if (x % i == 0 || x % (i + 2) == 0){
            return false;
        }
    }

    return true;
}


int storage_capacity(float d, float v){
    return (int)(d/v);
}

void swap(int *x, int *y) {
    if (!x || !y) return;

    int aux = *y;
    *y = *x;
    *x = aux;
}

int array_max(const int *array, int length) {
    if (!array) return false;

    int max = array[0];

    for (int i = 1; i < length; i++){
        if (array[i] > max){
            max = array[i];
        }
    }

    return max;

}

void array_map(int *array, int length, int f(int)) {
    if (!f || !array) return;
    for (int i = 0; i < length; i++){
        array[i] = f(array[i]);
    }
}

int *copy_array(const int *array, int length) {
    if (!array) return false;

    int * newArray = malloc(length * sizeof(int));

    if (!newArray) return false;

    for (int i = 0; i < length; i++){
        newArray[i] = array[i];
    }

    return newArray;
}

int **copy_array_of_arrays(const int **array_of_arrays, const int *array_lenghts, int array_amount){
    if (!array_of_arrays || !array_lenghts) return false;

    int ** newArray = malloc(array_amount * sizeof(int *));
    
    if (!newArray) return false;

    for (int i = 0; i < array_amount; i++){
        if (!array_of_arrays[i]){ 
            newArray[i] = NULL;
        }
        
        else{
            newArray[i] = copy_array(array_of_arrays[i], array_lenghts[i]);

            if (!newArray[i]){
                for (int j = 0; j <= i; j++){
                    free(newArray[j]);
                }

                free(newArray);
                return false;
            }
        }
    }


    return newArray;
}

void free_array_of_arrays(int **array_of_arrays, int *array_lenghts, int array_amount){
    if (!array_of_arrays || !array_lenghts) return;

    for (int i = 0; i < array_amount; i++){
        free(array_of_arrays[i]);
    }

    free(array_of_arrays);
    free(array_lenghts);
}

void bubble_sort(int *array, int length){
    if (!array) return;

    int i, j;
    bool swapped;
    for (i = 0; i < length - 1; i++) {
        swapped = false;
        for (j = 0; j < length - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
                swapped = true;
            }
        }

        if (!swapped) break;
    }
}

bool array_equal(const int *array1, int length1, const int *array2, int length2){
    if (!array1){
        if (!array2) return true;
        return false;
    }

    if (!array2) return false;

    if (!(length1 == length2)) return false;

    for (int i = 0; i < length1; i++){
        if (!(array1[i] == array2[i])) return false;
    }

    return true;
}

bool integer_anagrams(const int *array1, int length1, const int *array2, int length2){
    //wtf no anda bien

    if (array1 == NULL || array2 == NULL) return false;
    if (!(length1 == length2)) return false;
    if (array_equal(array1, length1, array2, length2)) return true;

    int * count1 = copy_array(array1, length1);
    int * count2 = copy_array(array2, length2);

    bubble_sort(count1, length1);
    bubble_sort(count2, length2);

    // for (int i = 0; i < 10; i++){
    //     count1[i] = 0;
    //     count2[i] = 0;
    // }

    // for (int j = 0; j < length1; j++){
    //     count1[array1[j]]++;
    //     count2[array2[j]]++;
    // }
    bool result = array_equal(count1, length1, count2, length2);
    
    free(count1);
    free(count2);

    return result;
}