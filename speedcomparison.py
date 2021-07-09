import time
import os, sys
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import colors


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/01_Sequential_MatMul")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/02_Parallel_Cuda_MatMul")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/03_Shared_Cuda_MatMul")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/04_CUBLAS_MatMul")

from sequential_matmul import sq_matmul_c
from cuda_arr import matMulSeqCuda
from cuda_arr_shared import matMulSeqCudaShared
from cudablas_matmul import matMulCudaBlas


def numpy_matmul_time(mtx_a, mtx_b, mtx_c=None):
    """
    """
    start_time = time.time()
    _ = mtx_a@mtx_b
    return time.time() - start_time

def sequential_cpp_time(mtx_a, mtx_b, mtx_c=None):
    """
    """
    start_time = time.time()
    _ = sq_matmul_c(mtx_a, mtx_b)
    return time.time() - start_time

# TODO: Sequential cuda

def sequential_cuda_sharedmem_time(mtx_a, mtx_b, mtx_c=None):
    """
    """
    start_time = time.time()
    _ = matMulSeqCudaShared(mtx_a, mtx_b, mtx_c)
    return time.time() - start_time

def cuda_blas_time(mtx_a, mtx_b, mtx_c=None):
    """
    """
    start_time = time.time()
    _ = matMulCudaBlas(mtx_a, mtx_b, mtx_c)
    return time.time() - start_time




if __name__ == '__main__':

    N_MIN = 4   # 32
    N_MAX = 16  # 4096

    increments=[]

    MAX_SEC = 0.05 # we allow for a second

    completion_times = { 'numpy_matmul_time':[],
                         'sequential_cpp_time':[],
                         'sequential_cuda_sharedmem_time':[],
                         'cuda_blas_time':[] }

    testing = [numpy_matmul_time,
               sequential_cpp_time,
               sequential_cuda_sharedmem_time,
               cuda_blas_time]

    for n in range(N_MIN, N_MAX):

        m = 2 << n-1
        n = 2 << n

        increments.append((m,n))

        a = np.random.random( (m, n) ).astype(np.float32)
        b = np.random.random( (n, m) ).astype(np.float32)
        c = np.zeros( (m, m) ).astype(np.float32)


        for func in testing:
            # we disqualify if the operation thakes longer than the limit
            curr_func_res = completion_times[func.__name__]
            if len(curr_func_res) > 1:
                if curr_func_res[-1] == None:
                    curr_func_res.append(None)
                elif curr_func_res[-1] <= MAX_SEC:
                    curr_func_res.append(func(a, b, c))
                else:
                    curr_func_res.append(None)
            else:
                curr_func_res.append(func(a, b, c))

    # plotting
    """
    increments = [max(i) for i in increments]
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    def plot_benchmark(dict_key, title, color):
        ax.plot(increments,
                completion_times[dict_key],
                marker='o',
                color=color)
        ax.set_title(title)

        for i, j in zip(increments, completion_times[dict_key]):
            if j: ax.annotate(str(round(j, 5)) , xy=(i + 0.0025, j + 0.0025))

    plot_benchmark('numpy_matmul_time', 'Numpy Matrix Product', '#1F766C')
    #plot_benchmark('sequential_cpp_time', 'C++ Sequential Matrix Product', '#DCA721')
    #plot_benchmark('sequential_cuda_sharedmem_time', 'Cuda Shared Memory Matrix Product', '#EF7512')
    #plot_benchmark('cuda_blas_time', 'Cuda Blas Matrix Product', '#CD3F1C')

    ax.set_xlabel('Matrix dimensions ' + r'$2^n * 2^{n-1}$')
    ax.set_ylabel('time(s)')
    ax.set_xscale('log', basex=2)

    fig.tight_layout()
    plt.grid()
    plt.xticks(increments)
    plt.show()
    """


    #calculate  percentages
    """
    all_vals = np.array(list(completion_times.values()))
    df = pd.DataFrame(all_vals)
    df.fillna(value=np.nan, inplace=True)

    incr = [*range(0, N_MAX - N_MIN)]

    for val in incr:
        df[val] = (df[val] / df[val].sum()) * 100
        quickest = df[val].idxmin()
        df[val] = df[val]/df[val][quickest]
        df[val] = df[val].round(1)

    print(df)
    """






