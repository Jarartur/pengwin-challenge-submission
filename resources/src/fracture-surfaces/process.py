from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from pathlib import Path

class Pengwin_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self, input_path, output_path, trained_model_path):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # FOR TESTING
        # root = '/home/ajurgas/Projects/PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/test'
        # set some paths and parameters
        self.input_path = input_path  # according to the specified grand-challenge interfaces
        self.output_path = output_path  # according to the specified grand-challenge interfaces
        self.trained_model_path = trained_model_path

    def predict(self):
        """
        Your algorithm goes here
        """
        print("nnUNet segmentation starting!")

        os.environ['nnUNet_compile'] = 'F'  # on my system the T does the test image in 2m56 and F in 3m15. Not sure if
        # 20s is worth the risk

        maybe_mkdir_p(self.output_path)

        # trained_model_path = '/home/ajurgas/Projects/PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/resources/models/fracture_seg/Dataset019_PENGWIN_fracture_surfaces2_masks/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'

        ct_mha, seg_mha = subfiles(self.input_path, suffix='.mha')
        # uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = Path(ct_mha).name[:-9]
        output_file_trunc = os.path.join(self.output_path, output_file_trunc)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(self.trained_model_path, use_folds=[0,1,2])
        predictor.dataset_json['file_ending'] = '.mha'

        # ideally we would like to use predictor.predict_from_files but this stupid docker container will be called
        # for each individual test case so that this doesn't make sense
        images, properties = SimpleITKIO().read_images([ct_mha, seg_mha])
        predictor.predict_single_npy_array(images, properties, None, output_file_trunc, True)

        print('Prediction finished')

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        # self.check_gpu() # waste of time. 10 mins is tight, yo
        print('Start prediction')
        self.predict()
        print('done')


if __name__ == "__main__":
    print("START")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    Pengwin_baseline(args.input, args.output, args.model).process()
