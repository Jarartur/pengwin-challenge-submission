import cupy as cp
import cupyx.scipy.ndimage as ndi

arr = cp.random.rand(1,1)
ndi.binary_dilation(arr)