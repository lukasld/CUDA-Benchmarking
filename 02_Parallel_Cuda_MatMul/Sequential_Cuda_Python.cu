#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
#define RANDOM_MN_RANGE 64

struct Matrix {
    int width;
    int height;
    // contiguously stored Matrix, in row first order
    float *elements;
};

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

    // runs for each col - row pair
    float tmpVal = 0;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = 0; i < A.width; ++i)
        tmpVal += A.elements[row * A.width + i] *
                  B.elements[i * B.width + col];
    C.elements[ row * C.width + col ] = tmpVal;
}

extern "C" {
    void mMul( Matrix *A, Matrix *B, Matrix *C ){

        Matrix d_A, d_B, d_C;

        // Matrix d_A
        d_A.width    =   A->width;
        d_A.height   =   A->height;
        size_t sizeA =   A->width * A->height * sizeof(float);
        // dynamically allocate cudaMemory for elemenst array
        cudaMalloc(&d_A.elements, sizeA);
        cudaMemcpy(d_A.elements, A->elements, sizeA, cudaMemcpyHostToDevice);

        // Matrix d_B
        d_B.width    =   B->width;
        d_B.height   =   B->height;
        size_t sizeB =   B->width * B->height * sizeof(float);
        // dynamically allocate cudaMemory for elemenst array
        cudaMalloc(&d_B.elements, sizeB);
        cudaMemcpy(d_B.elements, B->elements, sizeB, cudaMemcpyHostToDevice);

        // Matrix d_C
        d_C.width    =   C->width;
        d_C.height   =   C->height;
        size_t sizeC =   C->width * C->height * sizeof(float);

        // dynamically allocate cudaMemory for elemenst array
        cudaMalloc(&d_C.elements, sizeC);

        // 16 * 16 = 256 threads per block
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Blocks per grid
        dim3 dimGrid(B->width / dimBlock.x, A->height / dimBlock.y);

        // calling the Kernel
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        // copy results from result matrix C to the host again
        cudaMemcpy(C->elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);

        printf("A is %f\n", A->elements[0]);
        printf("B is %f\n", B->elements[0]);
        printf("C is %f\n", C->elements[0]);

        // free the cuda memory
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

    int mSize = 1<<4, nSize = 1<<6;

    // assign values to members height, width
    pmA->width = nSize, pmA->height = mSize;
    pmB->width = nSize, pmB->height = mSize;
    pmC->width = nSize, pmC->height = mSize;
    fillMatrix(pmA);
    fillMatrix(pmB);

    int nmSize = mSize * nSize;

    pmC->elements = (float*)calloc(nmSize, sizeof(float));
    MatMul(pmA, pmB, pmC);

    for (int i = 0; i < pmC->width * pmC->height; i++){
        printf("%f\n", pmC->elements[i]);
    }

    free(pmA);
    free(pmB);
    free(pmC);

    return 0;
}
*/



