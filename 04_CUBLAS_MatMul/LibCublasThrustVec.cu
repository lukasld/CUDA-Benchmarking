#include <cublas_v2.h>

#include <cstring>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

// this struct could be in a header shared h file
struct Matrix {
    int width;
    int height;
    float *elements;
};

extern "C" {

    void cuBlasMmul(Matrix *A, Matrix *B, Matrix *C){

        Matrix d_A, d_B, d_C;

        // Matrix d_A
        d_A.width    =   A->width;
        d_A.height   =   A->height;
        size_t sizeA =   A->width * A->height * sizeof(float);
        // dynamically allocate cudaMemory for elements array
        thrust::device_vector<float> dVecA(d_A.elements, d_A.elements + sizeA);

        // Matrix d_B
        d_B.width    =   B->width;
        d_B.height   =   B->height;
        size_t sizeB =   B->width * B->height * sizeof(float);
        // dynamically allocate cudaMemory for elements array
        thrust::device_vector<float> dVecB(d_B.elements, d_B.elements + sizeB);

        // Matrix d_C
        d_C.width    =   C->width;
        d_C.height   =   C->height;
        size_t sizeC =   C->width * C->height * sizeof(float);
        thrust::device_vector<float> dVecC(d_C.elements, d_C.elements + sizeC);

        // handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // scaling factor
        float alpha = 1.f;
        float beta = 0.f;

        // CUBLAS_OP_T -> normal matrix (matrix 1)
        // CUBLAS_OP_T -> transpose of the matrix (matrix 2)
        // q_rows_num, x_rows_num, dim -> m, n, k
        //  (m*n) * (n*k) = (m*k)
        // alpha? - beta?
        // q_device, x -> device pointer to q_device and x
        // q_rows_num -> lda -> leading dimension to a
        // x_rows_num -> ldb -> leading dimension to b
        // x_q_multiplication -> device pointer to c
        // q_rows_num -> leading dimension to c

        // height = rows, width = columns

        // COL-Major order
        // we can flip the order here instead of transposing it in numpy
        // m = number of rows (height) of matrices A and C
        // n = number of cols (width) of matrices B and C
        // k = number of cols (width) of matrices A and B
        
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N, // transa, transb
                    A->height, B->width, B->height, 
                    &alpha, // 1
                    thrust::raw_pointer_cast(&dVecA[0]), A->height,
                    thrust::raw_pointer_cast(&dVecB[0]), B->height,
                    &beta, // 0
                    thrust::raw_pointer_cast(&dVecC[0]), A->height);

        /*  
        for (int i = 0; i < ((int) sizeC) / 4; i++){
            printf( "%f\n", d_C.elements[i] );
        }
        */

        // copy results from result matrix C to the host again

        float *c_raw = thrust::raw_pointer_cast(&dVecC[0]);
        C->elements = c_raw;
        //cudaMemcpy(C->elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);

        // printf("A is %f\n", A->elements[0]);
        // printf("B is %f\n", B->elements[0]);
        // printf("C is %f\n", C->elements[0]);

        // synchronize
        cudaDeviceSynchronize();

        // free devce memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}


/*
void fillMatrix(Matrix *mX){
    // we have width * height elements in our matrix
    int mXsz = mX->height * mX->width;
    // we are allocating the values for float array
    mX->elements = (float*)malloc(sizeof(float) * mXsz);
    // we loop through the range of all elements
    for (int i = 0; i < mXsz; i++){
        // filling it with a random value between 0 and 1
        // mX->elements[i] = (float) rand()/RAND_MAX;
        mX->elements[i] = (float) 1.0;
    }
}

int main(){

    // start the random number generator
    srand((unsigned int)time(NULL));
    // allocating memory space to all three Matrices
    Matrix *pmA = (Matrix*) malloc(sizeof(Matrix));
    Matrix *pmB = (Matrix*) malloc(sizeof(Matrix));
    Matrix *pmC = (Matrix*) malloc(sizeof(Matrix));

    int mSize = 1<<4, nSize = 1<<4;

    // assign values to members height, width
    pmA->width = nSize, pmA->height = mSize;
    pmB->width = nSize, pmB->height = mSize;
    pmC->width = nSize, pmC->height = mSize;
    fillMatrix(pmA);
    fillMatrix(pmB);

    int nmSize = mSize * nSize;

    pmC->elements = (float*)calloc(nmSize, sizeof(float));

    cuBlasMmul(pmA, pmB, pmC);

    //printf("%f\n", pmC->elements[0]);

    for (int i = 0; i < pmC->width * pmC->height; i++){
        printf("%f\n", pmC->elements[i]);
    }

    free(pmA);
    free(pmB);
    free(pmC);

    return 0;
}
*/
