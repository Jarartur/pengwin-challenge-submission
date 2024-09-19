from collections import OrderedDict
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped

import torch
def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0, dtype_out = None,
                         cupy=True):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    if cupy:
        print('cupy!')
        import cupy as np
        # from scipy.ndimage.interpolation import map_coordinates
        from cupyx.scipy.ndimage import map_coordinates
        from cucim.skimage.transform import resize

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        mempool = np.get_default_memory_pool()
        pinned_mempool = np.get_default_pinned_memory_pool()

        # og_dtype = data.dtype
        # data = np.asarray(data, dtype=np.float16)
        data = np.asarray(data)
    else:
        import numpy as np
        from skimage.transform import resize
        from scipy.ndimage.interpolation import map_coordinates
        print('no cupy')
    print('start resampling')
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        def resize_segmentation(segmentation, new_shape, order=3):
            '''
            Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
            hot encoding which is resized and transformed back to a segmentation map.
            This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
            :param segmentation:
            :param new_shape:
            :param order:
            :return:
            '''
            new_shape = new_shape.tolist()
            tpe = segmentation.dtype
            unique_labels = np.unique(segmentation)
            assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
            if order == 0:
                return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
            else:
                reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
                for i, c in enumerate(unique_labels):
                    mask = segmentation == c
                    reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
                    reshaped[reshaped_multihot >= 0.5] = c
                return reshaped
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *(new_shape).tolist()), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                reshaped_here = np.zeros((data.shape[1], *new_shape_2d))
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                    else:
                        unique_labels = np.sort(np.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                        for i, cl in enumerate(unique_labels):
                            reshaped_final[c][np.round(
                                map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest')) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # print("no separate z, order", order)
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        print('resampling done')
        if cupy:
            reshaped_final = reshaped_final.get()
            # reshaped_final = reshaped_final.astype(og_dtype)
            gc.collect()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print('cupy memory free')

        return reshaped_final
    else:
        print("no resampling necessary")
        if cupy:
            data = data.get()
            # data = data.astype(og_dtype)
            gc.collect()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print('cupy memory free')

        return data
