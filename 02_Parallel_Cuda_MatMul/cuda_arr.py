import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *


class Matrix(Structure):
    _fields_ = [("width", c_int),
                ("height", c_int),
                ("elements", POINTER(c_float))]

libc = CDLL("/home/lukas/00_Projects/03_CudaMatmulComp/02_Parallel_Cuda_MatMul/Sequential_Cuda_Python.so")

libc.mMul.argtypes = [ POINTER(Matrix), POINTER(Matrix), POINTER(Matrix) ]
libc.mMul.restype = None

def npArrtoMatrixPtr(data: np.ndarray) -> (POINTER(Matrix), tuple):
    """
    numpy arr to Matrix pointer
    @return (pointer to arr, shape)
    """
    data = data.astype(np.float32)

    h, w = data.shape
    mXp = Matrix(w, h, data.ctypes.data_as(POINTER(c_float)))
    return (pointer(mXp), np.shape(data))

def matMulSeqCuda( npMa, npMb, npMc ):
    """
    multiplies mA with mB sequentually using mC
    """
    assert len(np.shape( npMa )) == 2 and \
           len(np.shape( npMb )) == 2 and \
           len(np.shape( npMc )) == 2, "Only 2D arrays accepted"

    pmA, szA = npArrtoMatrixPtr(npMa)
    pmB, szB = npArrtoMatrixPtr(npMb)
    pmC, szC = npArrtoMatrixPtr(npMc) # the resulting array

    libc.mMul( pmA, pmB, pmC )
    c = pmC.contents

    return np.ctypeslib.as_array(c.elements, shape=(c.height, c.width))


if __name__ == '__main__':

    a = np.ones( (64, 32) ).astype(np.float32)
    b = np.ones( (64, 32) ).astype(np.float32).T
    c = np.zeros( (64, 64) ).astype(np.float32)

    #a = np.random.random_sample((16, 8))
    #b = np.random.random_sample((16, 8))

    print(np.shape(a))
    print(np.shape(b))
    c_answ = matMulSeqCuda(a, b, c)
    print(np.shape(c_answ))
