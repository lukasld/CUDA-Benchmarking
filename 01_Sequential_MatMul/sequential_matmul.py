import os
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import * 


PARENTPTH = os.path.dirname(os.path.abspath(__file__))
SOPATH = PARENTPTH + "/sequential_matmul_c.so"


libc = CDLL(SOPATH)


class Arr(Structure):
    _fields_ = [ ('elems', POINTER(c_float)) ]


libc.mMultiply.argtypes = [ POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int ]
libc.mMultiply.restype = POINTER(c_float)


def npArrtoPtr(data: np.ndarray) -> (POINTER(c_float), tuple):
    """
    numpy arr to ctypes ptr
    @return (pointer to arr, shape)
    """
    c_float_p = POINTER(c_float)
    data = data.astype(np.float32)
    return (data.ctypes.data_as(c_float_p), np.shape(data))

def sq_matmul_c(mA, mB):
    """
    multiplies mA with mB sequentually using C
    """
    assert len(np.shape(mA)) == 2 and len(np.shape(mB)) == 2,  "Only 2D arrays accepted"
    assert np.shape(mA)[0] == np.shape(mB)[1], "Rows of mA need to be the same as Cols of mB"

    pmA, szA = npArrtoPtr(mA)
    pmB, szB = npArrtoPtr(mB)
    pmC = libc.mMultiply(pmA, pmB,
                         szA[0], szA[1],
                         szB[0], szB[1])
    return np.ctypeslib.as_array(pmC, shape=(szA[0], szB[1]))


'''
if __name__ == '__main__':


    a = np.ones( (60, 50) ).astype(np.float32)
    b = np.ones( (10, 60) ).astype(np.float32)

    print(np.shape(matrixMultiply(a, b)))
'''
