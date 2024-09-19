from pathlib import Path
import SimpleITK as sitk
import numpy as np

THRESHOLD = 0.002

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Threshold')
    parser.add_argument('--softmax', type=Path, help='Input image')
    parser.add_argument('--prediction', type=Path, help='Prediction image')
    args = parser.parse_args()

    print("Loading softmax...")
    softmax = np.load(args.softmax)
    softmax1 = softmax['probabilities'][0]
    softmax0 = softmax['probabilities'][1]

    print("Thresholding...")
    output = np.zeros_like(softmax0)
    output[softmax0 > THRESHOLD] = 1

    print("Saving...")
    output = sitk.GetImageFromArray(output)
    if args.prediction is not None:
        pred = sitk.ReadImage(args.prediction)
        output.CopyInformation(pred)
    sitk.WriteImage(output, str(args.softmax).replace('.npz', f'_thresholded{THRESHOLD}.mha'))
