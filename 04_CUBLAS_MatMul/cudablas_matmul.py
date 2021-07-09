import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *


class Matrix(Structure):
    _fields_ = [("height", c_int),
                ("width", c_int),
                ("elements", POINTER(c_float))]

#libc = CDLL("/home/lukas/00_C_Training/04_Python_C/02_Parallel_Cuda_MatMul/Sequential_Cuda_Python.so")
#libc = CDLL("/home/jn/02_C_Training/04_Python_C/02_Parallel_Cuda_MatMul/Sequential_Cuda_Python.so")
#libc = CDLL("/home/jn/02_C_Training/04_Python_C/04_CUBLAS_MatMul/LibCublas.so")

libc = CDLL("/home/lukas/00_C_Training/04_Python_C/04_CUBLAS_MatMul/LibCublas.so")


libc.cuBlasMmul.argtypes = [ POINTER(Matrix), POINTER(Matrix), POINTER(Matrix) ]
libc.cuBlasMmul.restype = None

def npArrtoMatrixPtr(data: np.ndarray) -> (POINTER(Matrix), tuple):
    """
    numpy arr to Matrix pointer
    @return (pointer to arr, shape)
    """
    # h = rows, w = columns
    h, w = data.shape

    mXp = Matrix(h, w, data.ctypes.data_as(POINTER(c_float)))
    return (pointer(mXp), np.shape(data))

def matMulCudaBlas( npMa, npMb, npMc ):
    """
    multiplies mA with mB sequentually using mC
    """
    assert len(np.shape( npMa )) == 2 and \
           len(np.shape( npMb )) == 2 and \
           len(np.shape( npMc )) == 2, "Only 2D arrays accepted"

    pmA, szA = npArrtoMatrixPtr(npMa)
    pmB, szB = npArrtoMatrixPtr(npMb)
    pmC, szC = npArrtoMatrixPtr(npMc) # the resulting array

    libc.cuBlasMmul( pmA, pmB, pmC )
    c = pmC.contents

    return np.ctypeslib.as_array(c.elements, shape=(c.height, c.width))


if __name__ == '__main__':

    c = np.zeros( (50000, 50000) ).astype(np.float32)
    #a = np.random.random_sample( (8, 4) ).astype(np.float32)
    #b = np.random.random_sample( (8, 4) ).astype(np.float32)
    a = np.ones((50000, 50000)).astype(np.float32).T
    b = np.ones((50000, 50000)).astype(np.float32)

    a_c = a.copy()
    b_c = b.copy()

    print('A')
    #print(a)
    print('\n')

    print('B')
    #print(b)
    print('\n')

    c_answ = matMulCudaBlas(a, b, c)

    print('C')
    #print(c_answ)
    print('\n')

    #np_res = a_c.T@b_c.T
    #print(np_res)
    #print(np_res == c_answ)
