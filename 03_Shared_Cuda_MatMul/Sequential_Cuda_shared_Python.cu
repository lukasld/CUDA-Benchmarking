#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
#define RANDOM_MN_RANGE 64

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix {
    int width;
    int height;
    int stride; 
    float* elements;
};

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to mak sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}


extern "C" {
    void MatMul( Matrix *A, Matrix *B, Matrix *C ){

        Matrix d_A, d_B, d_C;

        // Matrix d_A
        d_A.width    =   A->width;
        d_A.stride   =   A->width;
        d_A.height   =   A->height;
        size_t sizeA =   A->width * A->height * sizeof(float);
        // dynamically allocate cudaMemory for elemenst array
        cudaMalloc(&d_A.elements, sizeA);
        cudaMemcpy(d_A.elements, A->elements, sizeA, cudaMemcpyHostToDevice);

        // Matrix d_B
        d_B.width    =   B->width;
        d_B.stride   =   B->width;
        d_B.height   =   B->height;
        size_t sizeB =   B->width * B->height * sizeof(float);
        // dynamically allocate cudaMemory for elemenst array
        cudaMalloc(&d_B.elements, sizeB);
        cudaMemcpy(d_B.elements, B->elements, sizeB, cudaMemcpyHostToDevice);

        // Matrix d_C
        d_C.width    =   C->width;
        d_C.stride   =   C->width;
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


