import os
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import isfile
import torch
# from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.inference.segmentation_export import remove_component_and_save
from nnunet.training.model_restore import load_trained_model
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from nnunet.utilities.one_hot_encoding import to_one_hot


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
    print('cupy!')
    import cupy as np
    # from scipy.ndimage.interpolation import map_coordinates
    from cucim.skimage.transform import resize

    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    mempool = np.get_default_memory_pool()
    pinned_mempool = np.get_default_pinned_memory_pool()

    segmentation = np.asarray(segmentation)
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).get().astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        
        reshaped = reshaped.get()
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        print('cupy memory free')

        return reshaped

def predict_save_to_queue(preprocess_fn, list_of_lists, output_files, segs_from_prev_stage, classes, transpose_forward):
    errors_in = []

    for i, l in enumerate(list_of_lists):
        # try:
            print(l)
            output_file = output_files[i]
            d, _, dct = preprocess_fn(l)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".mha"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the my_multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            # if np.prod(d.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save, 4 because float32 is 4 bytes
            #     print(
            #         "This output is too large for python process-process communication. "
            #         "Saving output temporarily to disk")
            #     np.save(output_file[:-4] + ".npy", d)
            #     d = output_file[:-4] + ".npy"
            return output_file, (d, dct)
    #     except KeyboardInterrupt:
    #         raise KeyboardInterrupt
    #     except Exception as e:
    #         print("error in", l)
    #         print(e)
    # if len(errors_in) > 0:
    #     print("There were some errors in the following cases:", errors_in)
    #     print("These cases were ignored.")
    # else:
    #     print("This worker has ended successfully, no errors to report")


def preprocess_single(trainer, list_of_lists, output_files, segs_from_prev_stage=None,allow_downsample=False):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)
    #print("segs_from_prev_stage:",segs_from_prev_stage)
    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    if allow_downsample:
        preprocess_fn = trainer.preprocess_patient_allow_downsample
    else:
        preprocess_fn = trainer.preprocess_patient
    processes = predict_save_to_queue(preprocess_fn,list_of_lists,output_files,
                          segs_from_prev_stage,
                          classes,
                          trainer.plans['transpose_forward'])

    return processes

def free_memory(to_delete: list):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    for _var in to_delete:
        print(f"deleting {_var}")
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()

import time
def predict_single_case(model, input_image_dir, output_filename, do_tta=False, clean_up=False, save_npz=False, save_stl=0, allow_downsample=False, cascade=False):

    # make new folder and change file extension to nii.gz
    dr, f = os.path.split(output_filename)
    if len(dr) > 0:
        maybe_mkdir_p(dr)
    ### if the file format is not niigz, change to it.
    if not f.endswith(".mha"):
        f, _ = os.path.splitext(f)
        f = f + ".mha"
    cleaned_output_file = join(dr, f)


    torch.cuda.empty_cache()
    print("empty cuda cache...")
    # print(torch.cuda.memory_summary())
    trainer, params = load_trained_model(model)   
    print("begin preprocessing...")
    segs_from_prev_stage = None if not cascade else str(Path(output_filename).parent/'3d_lowres'/Path(output_filename).name)
    preprocessing = preprocess_single(trainer, [[input_image_dir]], [cleaned_output_file], [segs_from_prev_stage], allow_downsample=allow_downsample)
    output_filename, (d, dct) = preprocessing
    # print(f"d: {d.shape}, dct: {dct}")
    if isinstance(d, str):
        data = np.load(d)
        os.remove(d)
        d = data

    print("begin prediction...")
    trainer.load_checkpoint_ram(params, False)
    trainer.data_aug_params['mirror_axes'] = (0, 1, 2)
    softmax=trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'],
                                                                     use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=False, mixed_precision=True)[1]
                                                                    # False, 1, trainer.data_aug_params['mirror_axes'],
                                                                    # True,True, 2 ,trainer.patch_size,True)
    transpose_forward = trainer.plans.get('transpose_forward')
    if transpose_forward is not None:
        transpose_backward = trainer.plans.get('transpose_backward')
        softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

    if save_npz:
        npz_file = output_filename[:-4] + ".npz"
    else:
        npz_file = None

    # trainer, params, transpose_forward, preprocessing = None, None, None, None
    # del trainer
    # del params
    # del transpose_forward
    # del preprocessing
    # gc.collect()
    # time.sleep(1)
    # torch.cuda.empty_cache()
    # print("empty cuda cache...")
    # time.sleep(1)

    free_memory(['trainer', 'params', 'transpose_forward', 'preprocessing'])
    time.sleep(1)
    
    # del trainer
    # torch.cuda.empty_cache()
    # print("empty cuda cache...")
    # print(torch.cuda.memory_summary())
    return remove_component_and_save(softmax, output_filename, dct, 1, None, None, None, npz_file, rm_components=clean_up, save_stl=save_stl)

if __name__ == "__main__":
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.mha where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-m', "--model_folder", type=Path, required=True, help="")
    args = parser.parse_args()
    file_name = list(Path(args.input_folder).glob('*.mha'))[0]
    output_file_name_3dlowres = os.path.join(args.output_folder, '3d_lowres', f'{file_name.name[:-9]}.mha')
    # Stage 1 - lowres
    model_folder = args.model_folder / "nnUNet" / "3d_lowres" / "Task013_pengwinraw" / "nnUNetTrainerV2__nnUNetPlansv2.1" / "fold_0"
    predict_single_case(str(model_folder), file_name, output_file_name_3dlowres)
#
    # Stage 2 - cascade
    model_folder = args.model_folder / "nnUNet" / "3d_cascade_fullres" / "Task013_pengwinraw" / "nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1" / "fold_0"
    output_file_name_cascade = os.path.join(args.output_folder, f'{file_name.name[:-9]}.mha')
    predict_single_case(str(model_folder), file_name, output_file_name_cascade, cascade=True)
