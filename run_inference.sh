python /opt/app/resources/src/move_file.py \
    /input/images/pelvic-fracture-ct \
    /tmp/images/pelvic-fracture-ct-segmentation/input \
    _0000

mkdir /tmp/images/pelvic-fracture-ct-segmentation/anatomical
# nnUNet_predict -m 3d_cascade_fullres -f 0 --disable_tta -t 13 \
#     -i /tmp/images/pelvic-fracture-ct-segmentation/input \
#     -o /tmp/images/pelvic-fracture-ct-segmentation/anatomical

python /opt/app/resources/src/fracSegNet/inference/predict_single.py \
    -i /tmp/images/pelvic-fracture-ct-segmentation/input \
    -o /tmp/images/pelvic-fracture-ct-segmentation/anatomical \
    -m /opt/app/resources/models/anatomical_seg \

python /opt/app/resources/src/move_file.py \
    /tmp/images/pelvic-fracture-ct-segmentation/anatomical \
    /tmp/images/pelvic-fracture-ct-segmentation/input \
    _0001

mkdir /tmp/images/pelvic-fracture-ct-segmentation/fractures
# nnUNetv2_predict -c 3d_fullres -d 19 -f 0 --disable_tta --save_probabilities -p nnUNetResEncUNetLPlans \
#     -i /tmp/images/pelvic-fracture-ct-segmentation/input \
#     -o /tmp/images/pelvic-fracture-ct-segmentation/fractures

python /opt/app/resources/src/fracture-surfaces/process.py \
    --input /tmp/images/pelvic-fracture-ct-segmentation/input \
    --output /tmp/images/pelvic-fracture-ct-segmentation/fractures \
    --model /opt/app/resources/models/fracture_seg/Dataset019_PENGWIN_fracture_surfaces2_masks/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres
    # --model /opt/app/resources/models/fracture_seg/Dataset020_PENGWIN_extended_fracture_surfaces2_masks/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres

python /opt/app/resources/src/fracture-surfaces/validation_pipeline_surfaces.py \
    --predictions /tmp/images/pelvic-fracture-ct-segmentation/fractures \
    --anatomical-seg /tmp/images/pelvic-fracture-ct-segmentation/anatomical \
    --output /output/images/pelvic-fracture-ct-segmentation