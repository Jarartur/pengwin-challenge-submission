from pathlib import Path
# import numpy as np
import cupy as np
import SimpleITK as sitk
# import scipy.ndimage as ndi
import cupyx.scipy.ndimage as ndi
from tqdm import tqdm

SOFTMAX_EXT = ".npz"
PRED_EXT = ".mha"
ANATOMICAL_SEG_EXT = ".mha"

# ANATOMICAL_LABELS = {
#     0: 'background',
#     1: 'sacrum',
#     11: 'left_hipbone',
#     21: 'right_hipbone'
# }
THRESHOLD = 1e-03

FRACTURE_DILATION_ITERS = 1 # this defines the number of voxels we can close at max during pre-processing
FRACTURE_CLOSING_ITERS = 10 # this defines the number of voxels we can close at max during pre-processing
NEIGHBORHOOD_DILATION_ITERS = 10 # this definies the neighborhood size of the fracture surface which will be post-processed
CLOSING_ITERS = NEIGHBORHOOD_DILATION_ITERS # this defines the number of voxels we can close at max during post-processing
POST_DILATION_ITERS = 50 + FRACTURE_DILATION_ITERS # + FRACTURE_PREPROCESS_ITERS + CLOSING_ITERS # this definies how many voxels we can grow the labels after the anatomical match and should be chosen according to the thickness of fracture surfaces (dilation is still limited by the anatomical seg)

STRUCT_ELEM_MORPH = ndi.generate_binary_structure(3, 1)
STRUCT_ELEM_LABEL = ndi.generate_binary_structure(3, 1)

FRACTURE_THRESHOLD = {
    1: 470,
    11: 750,
    21: 750,
}

DETECTION_THRESHOLD = 200

SMALL_FRAC_DIST_LIMIT = 10
BIG_FRAC_DIST_LIMIT = 60

def get_thresholded_image(softmax_path: Path|str, threshold: float = 0.0001) -> np.ndarray:
    softmax = np.load(softmax_path)
    softmax0 = softmax['probabilities'][1]

    output = np.zeros_like(softmax0)
    output[softmax0 > threshold] = 1
    return output

def create_image_with_metadata(image: np.ndarray, other_file: Path|str) -> sitk.Image:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(other_file))
    reader.ReadImageInformation()
    
    img = sitk.GetImageFromArray(image.get().astype(np.uint8))
    img.SetDirection(reader.GetDirection())
    img.SetOrigin(reader.GetOrigin())
    img.SetSpacing(reader.GetSpacing())
    return img


def extract_edges(img: np.ndarray) -> np.ndarray:
    labels = np.unique(img)
    edges_mask = np.zeros_like(img)
    for label in labels:
        if label == 0:
            continue
        mask = img == label
        edges = ndi.binary_dilation(mask) ^ mask # risk of overlap but it shouldn't bother us while it helps with reducing closing of think bone sections.
        edges_mask[edges] = 1
    return edges_mask

def preprocess_prediction(fracture_surface: np.ndarray):
    if FRACTURE_DILATION_ITERS > 0:
        fracture_surface = ndi.binary_dilation(fracture_surface, structure=STRUCT_ELEM_MORPH, iterations=FRACTURE_DILATION_ITERS, brute_force=True)
    fracture_surface = ndi.binary_closing(fracture_surface, structure=STRUCT_ELEM_MORPH, iterations=FRACTURE_CLOSING_ITERS, brute_force=True)
    return fracture_surface

def merge_edges(anatomical_edges: np.ndarray, fracture_surface: np.ndarray):
    # merge edges
    merged = np.logical_or(anatomical_edges, fracture_surface)
    # make dilated mask of fracture surface
    neighborhood_mask = ndi.binary_dilation(fracture_surface, structure=STRUCT_ELEM_MORPH, iterations=NEIGHBORHOOD_DILATION_ITERS, brute_force=True)
    # close the merged edges with dilated mask
    # closed = ndi.binary_closing(merged, structure=STRUCT_ELEM_MORPH, iterations=CLOSING_ITERS, mask=neighborhood_mask) leaves holes at the edges for some reason
    processed = ndi.binary_closing(merged, structure=STRUCT_ELEM_MORPH, iterations=CLOSING_ITERS, brute_force=True)
    merged[neighborhood_mask] = processed[neighborhood_mask]
    return merged

def convert_anatomical_seg(anatomical_seg: np.ndarray):
    # anatomical_seg = np.where(anatomical_seg == 2, 11, anatomical_seg)
    anatomical_seg[anatomical_seg == 2] = 11
    # anatomical_seg = np.where(anatomical_seg == 3, 21, anatomical_seg)
    anatomical_seg[anatomical_seg == 3] = 21
    anatomical_seg[anatomical_seg == 4] = 0
    return anatomical_seg

def remove_edges_from_labels(image_labeled:np.ndarray, edges:np.ndarray):
    labels_iou = {}
    for label in np.unique(image_labeled):
        labels_iou[label] = calculate_3d_iou(image_labeled==label, edges)
    edges_label = max(labels_iou, key=labels_iou.get)

    return np.where(image_labeled != edges_label, image_labeled, 0)

def label(image: np.ndarray, anatomical_binary: np.ndarray):
    image_inv = 1 - image

    image_labeled, n_labels = ndi.label(image_inv, structure=STRUCT_ELEM_LABEL)
    # image_labeled = np.where(image_labeled > 1, image_labeled, 0) # to merge background label with edges
                                                                  # TODO: make sure edges are always labeled as 1
                                                                  # using remove_edges_from_labels
    image_labeled[image_labeled <= 1] = 0 # same us above
    image_labeled[~anatomical_binary] = 0 # removes labels in the background (outside of anatimical segmentaiton)
    
    return image_labeled

# def calculate_3d_iou(vol1: np.ndarray, vol2: np.ndarray) -> float:
#     # Ensure the inputs are boolean arrays to optimize logical operations
#     vol1 = vol1.astype(bool)
#     vol2 = vol2.astype(bool)
   
#     # Calculate intersection and union using np.count_nonzero
#     intersection = np.count_nonzero(np.logical_and(vol1, vol2))
#     union = np.count_nonzero(np.logical_or(vol1, vol2))
   
#     # Calculate IoU
#     if union == 0:
#         return 0.0  # Avoid division by zero
#     return intersection / union

def calculate_3d_iou(vol1: np.ndarray, vol2: np.ndarray):
    # Calculate intersection and union
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()
    # Calculate IoU
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def find_anatomical_match(label_mask: np.ndarray, anatomical_seg: np.ndarray):
    matches = {}
    for i in [0, 1, 11, 21]: # 0 is for deleting small labels that are not in the anatomical segmentation
        anatomical_mask = anatomical_seg == i
        iou = calculate_3d_iou(label_mask, anatomical_mask)
        matches[i] = iou
    return max(matches, key=matches.get)

def find_closest_fractures(labels, image_labeled, fractures_mask, anatomical_match):
    # mask = np.logical_and(fractures_mask<anatomical_match, fractures_mask>anatomical_match+9)
    # fractures_mask[mask] = 0
    labeled_mask = np.logical_and(fractures_mask>=anatomical_match, fractures_mask<=anatomical_match+9)

    indices = ndi.distance_transform_edt(~labeled_mask, return_distances=False, return_indices=True)

    nearest_labels = []
    for label in labels:
        unlabeled_mask = image_labeled == label
        unlabeled_centroid = np.array(ndi.center_of_mass(unlabeled_mask))
        centroid_coords = tuple(unlabeled_centroid.round().astype(int))
        nearest_labels += [fractures_mask[indices[0][centroid_coords], indices[1][centroid_coords], indices[2][centroid_coords]]]

    return nearest_labels

def find_closest_fractures2(labels, image_labeled, fractures_mask, anatomical_match):
    # mask = np.logical_and(fractures_mask<anatomical_match, fractures_mask>anatomical_match+9)
    # fractures_mask[mask] = 0
    labeled_mask = np.logical_and(fractures_mask>=anatomical_match, fractures_mask<=anatomical_match+9)

    distances, indices = ndi.distance_transform_edt(~labeled_mask, return_distances=True, return_indices=True)

    nearest_labels = []
    for label in labels:
        unlabeled_mask = image_labeled == label
        unlabeled_centroid = np.array(ndi.center_of_mass(unlabeled_mask))
        centroid_coords = tuple(unlabeled_centroid.round().astype(int))
        # if np.min(distances[unlabeled_mask]) < SMALL_FRAC_DIST_LIMIT:
        if (distances[centroid_coords]) < SMALL_FRAC_DIST_LIMIT:
            nearest_labels += [fractures_mask[indices[0][centroid_coords], indices[1][centroid_coords], indices[2][centroid_coords]]]
        else:
            nearest_labels += [0]
    return nearest_labels

def process_label(label_mask, anatomical_segmentation, prediction_edges, anatomical_match):
    dilation_mask = anatomical_segmentation == anatomical_match
    # dilation_mask = np.where(prediction_edges>0, 0, anatomical_seg)
    # dilation_mask[prediction_edges>0] = 0

    # out = ndi.binary_dilation(label_mask, structure=STRUCT_ELEM_MORPH, iterations=-1, mask=dilation_mask)
    out = ndi.binary_dilation(label_mask, structure=STRUCT_ELEM_MORPH, iterations=POST_DILATION_ITERS, mask=dilation_mask, brute_force=True) # to compensate for the thickness of the fracture surface
    return out

def process_output(fracture_mask, anatomical_segmentation, anatomical_match):
    # getting all the fractures as binary mask
    mask = np.logical_and(fracture_mask>=anatomical_match, fracture_mask<=anatomical_match+9)
    # mask = (mask != 0)

    # getting the indicies of nearest fracture for each pixel
    output = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    # assigning every voxel outside of fractures a closest fracture label
    output = fracture_mask[tuple(output)]

    # restricting the output to the anatomical segmentation while allowing the fractures to grow inside the anatomical segmentation by POST_DILATION_ITERS iterations
    dilation_mask = anatomical_segmentation == anatomical_match
    output_mask = ndi.binary_dilation(mask, structure=STRUCT_ELEM_MORPH, iterations=POST_DILATION_ITERS, mask=dilation_mask, brute_force=True)
    return output_mask * output

def process_output2(fracture_mask, anatomical_segmentation, anatomical_match):
    # getting all the fractures as binary mask
    if anatomical_match is None:
        anatomical_submask = fracture_mask>0
    else:
        anatomical_submask = np.logical_and(fracture_mask>=anatomical_match, fracture_mask<=anatomical_match+9)

    # getting the indicies of nearest fracture for each pixel
    output = ndi.distance_transform_edt(~anatomical_submask, return_distances=False, return_indices=True)
    # assigning every voxel outside of fractures a closest fracture label
    output = fracture_mask[tuple(output)]

    # restricting the output to the anatomical segmentation while allowing the fractures to grow inside the anatomical segmentation by POST_DILATION_ITERS iterations
    if anatomical_match is None:
        dilation_mask = anatomical_segmentation>0
    else:
        dilation_mask = anatomical_segmentation == anatomical_match
    dilation_iters = POST_DILATION_ITERS if anatomical_match is not None else POST_DILATION_ITERS//2
    output_mask = ndi.binary_dilation(anatomical_submask, structure=STRUCT_ELEM_MORPH, iterations=dilation_iters, mask=dilation_mask, brute_force=True)
    fracture_mask[dilation_mask] = output_mask[dilation_mask] * output[dilation_mask]
    return fracture_mask

def filter_far_labels(fracture_mask, anatomical_match, anatomical_segmentation=None):
    
    anatomical_submask = np.logical_and(fracture_mask>=anatomical_match, fracture_mask<=anatomical_match+9) # getting all the fractures as binary mask
    present_labels = np.unique(fracture_mask[anatomical_submask])
    if len(present_labels) > 2: # if there is only background (0) and one label (1,11,21) then we musn't filter as the distance will be infinite
        for label in range(anatomical_match, anatomical_match+10):# ommiting 0
            mask = fracture_mask==label
            
            if (mask).any():
                anatomical_submask = np.logical_and(fracture_mask>=anatomical_match, fracture_mask<=anatomical_match+9) # getting all the fractures as binary mask
                anatomical_submask[mask] = 0 # removing the current fracture label
                distances = ndi.distance_transform_edt(~anatomical_submask, return_distances=True, return_indices=False)

                if np.min(distances[mask]) > BIG_FRAC_DIST_LIMIT:
                    fracture_mask[mask] = 0

                    if anatomical_segmentation is not None:

                        anat_mask = anatomical_segmentation > 0
                        anat_mask[mask] = 0
                        distances, indicies = ndi.distance_transform_edt(~anat_mask, return_distances=True, return_indices=True)
                        
                        if np.min(distances[mask]) < 3:
                            output = fracture_mask[tuple(indicies)]
                            fracture_mask[mask] = output[mask]

                        # dilation_iters = POST_DILATION_ITERS//2
                        # output_mask = ndi.binary_dilation(mask, structure=STRUCT_ELEM_MORPH, iterations=dilation_iters, mask=dilation_mask, brute_force=True)


    return fracture_mask

def assign_and_grow_labels(image_labeled: np.ndarray, anatomical_segmentation: np.ndarray, prediction_edges: np.ndarray):
    labels_counter = {
        1: 0,
        11: 0,
        21: 0,
    }

    output = np.zeros_like(image_labeled)
    
    queue = np.unique(image_labeled)[1:].tolist() # starting from 1: to omit 0 (background class)
    queue = sorted(queue, key=lambda x: -np.sum(image_labeled==x)) # sort this so that the biggest fragments are processed first
    small_labels = {1:[], 11:[], 21:[]}
    print(f"Processing queue: {queue}")
    for i in (pbar:=tqdm(queue)):
        label_mask = image_labeled == i
        anatomical_match = find_anatomical_match(label_mask, anatomical_segmentation)
        pbar.set_description(f"Anatomical match for label {i}: {anatomical_match}")

        if anatomical_match == 0:
            # if we work with external edges then this should also eliminate the edges' label
            # should be fixed by now but it doesn't cost us anything so I will leave it.
            continue # Esentially delete the label that falls into the background main segmentation
        if labels_counter[anatomical_match]>9:
            small_labels[anatomical_match].append(i) # if we sort first then if there are more than 9 labels they are gonna be small anyway
            continue # Preventing too many labels for the same anatomical structure
        if label_mask.sum() < DETECTION_THRESHOLD:
            continue # really small labels are not considered
        if label_mask.sum() < FRACTURE_THRESHOLD[anatomical_match]:
            # label = anatomical_match+0 # Preventing too small labels from being assigned a whole new fragment
                                         # instead we assign it to the main fragment
            small_labels[anatomical_match].append(i)
            continue

        # out = process_label(label_mask, anatomical_segmentation, prediction_edges, anatomical_match)

        label = anatomical_match+labels_counter[anatomical_match]
        labels_counter[anatomical_match] += 1
        
        output[label_mask] = label

    for anatomical_match in (pbar:=tqdm([1, 11, 21])):
        pbar.set_description(f"1st Processing anatomical match: {anatomical_match}")
        output = process_output2(output, anatomical_segmentation, anatomical_match)

    for anatomical_match in (pbar:=tqdm(small_labels)):
        pbar.set_description(f"Fixing small detections for: {anatomical_match}")
        nearest_fractures = find_closest_fractures2(small_labels[anatomical_match], image_labeled, output, anatomical_match)
        for i, nearest_fracture in zip(small_labels[anatomical_match], nearest_fractures):
            label_mask = image_labeled == i
            # out = process_label(label_mask, anatomical_segmentation, prediction_edges, anatomical_match)
            output[label_mask] = nearest_fracture

    for anatomical_match in (pbar:=tqdm([1, 11, 21])):
        pbar.set_description(f"3rd Processing anatomical match: {anatomical_match}")
        output = process_output2(output, anatomical_segmentation, anatomical_match)
    
    for anatomical_match in (pbar:=tqdm([1, 11, 21])):
        pbar.set_description(f"Filtering far away labels for anatomical match: {anatomical_match}")
        output = filter_far_labels(output, anatomical_match, anatomical_segmentation)

    # print(f"Final labels growing")
    # output = process_output2(output, anatomical_segmentation, None)

    # output = process_output(output, anatomical_segmentation)
    return output

def get_anatomical_seg(predictions_path: Path, anatomical_seg_path: Path, case_file: Path):
    # anatomical_seg_file = anatomical_seg_path / case_file.relative_to(predictions_path).with_suffix(ANATOMICAL_SEG_EXT)
    anatomical_seg_file = (anatomical_seg_path / case_file.name).with_suffix(ANATOMICAL_SEG_EXT)

    anatomical_seg = sitk.ReadImage(anatomical_seg_file)
    anatomical_seg = np.array(sitk.GetArrayFromImage(anatomical_seg))
    return anatomical_seg, anatomical_seg_file

def save_debug_image(name:str, image: np.ndarray, metadata_img: Path, save_path: Path):
    image = create_image_with_metadata(image, metadata_img)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(image, save_path.with_suffix(f'.{name}.debug.nii.gz'))

if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser(description='Validation Pipeline')
    parser.add_argument('--predictions', type=Path, help='Results folder')
    parser.add_argument('--anatomical-seg', type=Path, help='First step folder')
    parser.add_argument('--output', type=Path, help='Output folder')
    args = parser.parse_args()

    print(args)

    for img in (pbar := tqdm(args.predictions.rglob(f"*{SOFTMAX_EXT}"))):
        pbar.set_description(f"Processing {img.name}")

        threshold_image = get_thresholded_image(img, THRESHOLD)
        threshold_processed_image = preprocess_prediction(threshold_image)

        anatomical_pred, anatomical_seg_file = get_anatomical_seg(args.predictions, args.anatomical_seg, img)

        anatomical_seg = convert_anatomical_seg(anatomical_pred)

        merged_image = merge_edges(extract_edges(anatomical_seg), threshold_processed_image)

        prediction_labels = label(merged_image, anatomical_seg>0)

        prediction = assign_and_grow_labels(prediction_labels, anatomical_seg, threshold_image)

        image = create_image_with_metadata(prediction, anatomical_seg_file)

        save_path = (args.output / img.name).with_suffix('.mha')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(image, save_path)