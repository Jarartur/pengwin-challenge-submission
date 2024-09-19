from pathlib import Path
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi

SOFTMAX_EXT = ".npz"
PRED_EXT = ".mha"
ANATOMICAL_SEG_EXT = ".mha"

# ANATOMICAL_LABELS = {
#     0: 'background',
#     1: 'sacrum',
#     11: 'left_hipbone',
#     21: 'right_hipbone'
# }

CLOSING_ITERS = 5
DILATION_ITERS = 1
EROSION_ITERS = 0

STRUCT_ELEM_MORPH = ndi.generate_binary_structure(3, 1)
STRUCT_ELEM_LABEL = ndi.generate_binary_structure(3, 1)

FRACTURE_THRESHOLD = 1000

def get_thresholded_image(softmax_path: Path|str, threshold: float = 0.0001) -> np.ndarray:
    softmax = np.load(softmax_path)
    softmax1 = softmax['probabilities'][0]
    softmax0 = softmax['probabilities'][1]

    output = np.zeros_like(softmax0)
    output[softmax0 > threshold] = 1
    return output

def create_image_with_metadata(image: np.ndarray, other_file: Path|str) -> sitk.Image:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(other_file))
    reader.ReadImageInformation()
    
    img = sitk.GetImageFromArray(image.astype(np.uint8))
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
        edges = ndi.binary_erosion(mask) ^ mask
        edges_mask[edges] = 1
    return edges_mask

def merge_edges(edges1: np.ndarray, edges2: np.ndarray):
    return np.logical_or(edges1, edges2)

def convert_anatomical_seg(anatomical_seg: np.ndarray):
    anatomical_seg = np.where(anatomical_seg == 2, 11, anatomical_seg)
    anatomical_seg = np.where(anatomical_seg == 3, 21, anatomical_seg)
    return anatomical_seg

def remove_edges_from_labels(image_labeled:np.ndarray, edges:np.ndarray):
    labels_iou = {}
    for label in np.unique(image_labeled):
        labels_iou[label] = calculate_3d_iou(image_labeled==label, edges)
    edges_label = max(labels_iou, key=labels_iou.get)

    return np.where(image_labeled != edges_label, image_labeled, 0)

def label(image: np.ndarray):
    if CLOSING_ITERS > 0:
        image = ndi.binary_closing(image, structure=STRUCT_ELEM_MORPH, iterations=CLOSING_ITERS)
    if DILATION_ITERS > 0:
        image = ndi.binary_dilation(image, structure=STRUCT_ELEM_MORPH, iterations=DILATION_ITERS)
    if EROSION_ITERS > 0:
        image = ndi.binary_erosion(image, structure=STRUCT_ELEM_MORPH, iterations=EROSION_ITERS)

    image_inv = 1 - image

    image_labeled, n_labels = ndi.label(image_inv, structure=STRUCT_ELEM_LABEL)
    image_labeled = np.where(image_labeled > 1, image_labeled, 0) # to merge background label with edges
                                                                  # TODO: make sure edges are always labeled as 1
                                                                  # using remove_edges_from_labels
    
    return image_labeled, image

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

def assign_and_grow_labels(image_labeled: np.ndarray, anatomical_segmentation: np.ndarray, prediction_edges: np.ndarray):
    labels_counter = {
        1: 0,
        11: 0,
        21: 0,
    }

    output = np.zeros_like(image_labeled)
    
    queue = np.unique(image_labeled)[1:].tolist() # starting from 1: to omit 0 (background class)
    queue = sorted(queue, key=lambda x: -np.sum(image_labeled==x)) # sort this so that the biggest fragments are processed first
    print(f"Processing queue: {queue}")
    for i in queue:
        label_mask = image_labeled == i
        anatomical_match = find_anatomical_match(label_mask, anatomical_segmentation)
        print(f"Found anatomical match for label {i}: {anatomical_match}")

        if anatomical_match == 0:
            continue # Esentially delete the label that falls into the background main segmentation
        if labels_counter[anatomical_match]>9:
            continue # Preventing too many labels for the same anatomical structure

        anatomical_seg = anatomical_segmentation == anatomical_match
        dilation_mask = np.where(prediction_edges>0, 0, anatomical_seg)

        out = ndi.binary_dilation(label_mask, structure=STRUCT_ELEM_MORPH, iterations=DILATION_ITERS+10, mask=dilation_mask) #TODO: replace with nearest-neighbor interpolation+dilation (https://stackoverflow.com/questions/12747319/scipy-label-dilation)
        out = ndi.binary_dilation(out, structure=STRUCT_ELEM_MORPH, iterations=5, mask=anatomical_seg) # we work on internal edges so 1 dilation to get things as original
        
        if label_mask.sum() < FRACTURE_THRESHOLD:
            label = anatomical_match+0 # Preventing too small labels from being assigned a whole new fragment
                                       # instead we assign it to the main fragment
        else:
            label = anatomical_match+labels_counter[anatomical_match]
            labels_counter[anatomical_match] += 1
        
        output[out] = label

    return output

def get_anatomical_seg(predictions_path: Path, anatomical_seg_path: Path, case_file: Path):
    # anatomical_seg_file = anatomical_seg_path / case_file.relative_to(predictions_path).with_suffix(ANATOMICAL_SEG_EXT)
    anatomical_seg_file = (anatomical_seg_path / case_file.name).with_suffix(ANATOMICAL_SEG_EXT)

    anatomical_seg = sitk.ReadImage(anatomical_seg_file)
    anatomical_seg = sitk.GetArrayFromImage(anatomical_seg)
    return anatomical_seg

if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser(description='Validation Pipeline')
    parser.add_argument('--predictions', type=Path, help='Results folder')
    parser.add_argument('--anatomical-seg', type=Path, help='First step folder')
    parser.add_argument('--output', type=Path, help='Output folder')
    parser.add_argument('--threshold', type=float, help='Threshold value')
    parser.add_argument('--debug', action='store_true', help='Save intermediary images')
    args = parser.parse_args()

    print(args)

    for img in (pbar := tqdm(args.predictions.rglob(f"*{SOFTMAX_EXT}"))):
        pbar.set_description(f"Processing {img.name}")

        prediction_image = get_thresholded_image(img, args.threshold)

        anatomical_pred = get_anatomical_seg(args.predictions, args.anatomical_seg, img)
        anatomical_seg = convert_anatomical_seg(anatomical_pred)
        prediction_image = merge_edges(extract_edges(anatomical_seg), prediction_image)

        prediction_labels, debug_image = label(prediction_image)
        prediction = assign_and_grow_labels(prediction_labels, anatomical_seg, prediction_image)

        image = create_image_with_metadata(prediction, img.with_suffix(PRED_EXT))

        save_path = (args.output / img.relative_to(args.predictions)).with_suffix('.pred.nii.gz')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(image, save_path)

        if args.debug:
            save_path = (args.output / img.relative_to(args.predictions).parent / 'debug' / img.name).with_suffix('.threshold.debug.nii.gz')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            prediction_image = create_image_with_metadata(prediction_image, img.with_suffix(PRED_EXT))
            sitk.WriteImage(prediction_image, save_path)

            save_path = (args.output / img.relative_to(args.predictions).parent / 'debug' / img.name).with_suffix('.edges.debug.nii.gz')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            debug_image = create_image_with_metadata(debug_image, img.with_suffix(PRED_EXT))
            sitk.WriteImage(debug_image, save_path)

            save_path = (args.output / img.relative_to(args.predictions).parent / 'debug' / img.name).with_suffix('.labels.debug.nii.gz')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            prediction_labels = create_image_with_metadata(prediction_labels, img.with_suffix(PRED_EXT))
            sitk.WriteImage(prediction_labels, save_path)