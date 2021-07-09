#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *elems;
} Arr;

Arr * arrMake(int randNrSz, int arrLen){

    Arr *pArr = malloc(sizeof(Arr));
    pArr->elems = malloc(sizeof(float) * randNrSz);

    for (int i = 0; i < arrLen; i++){
        float rand_val = (float) (rand() % randNrSz);
        pArr->elems[i] = rand_val;
        //printf("%.3f\n", rand_val);
    }
    return pArr;
}


float *mMultiply (float *mA, float *mB, int mDimA, int nDimA, int mDimB, int nDimB){
    /*
     * sequential implementation of matrix product
     */

    float *mC = malloc(sizeof(float) * mDimA * nDimB);
    for (int i = 0; i < mDimA; ++i) {
        for (int j = 0; j < nDimB; ++j) {
            float sum = 0.0;
            for (int k = 0; k < mDimB; k++)
                sum = sum + mA[i * nDimA + k] * mB[k * nDimB + j];
            mC[i * nDimB + j] = sum;
        }
    }

    return mC;
}



/*
int main(){
    int testVal = 5;
    ArrSz * a = arrTest(5);
    for (int i = 0; i < a->len; i++){
        printf("%.3f\n", *(a->arr + i));
    }
    //printf("%d\n", *(a->pt));
}
*/

