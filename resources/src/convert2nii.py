import SimpleITK as sitk
from pathlib import Path

def convert(input_path: Path, output_path: Path):
    input_path = list(input_path.glob('*.mha'))[0]
    output_path = output_path / (f'{input_path.stem}_0000' + '.nii.gz')
    sitk.WriteImage(sitk.ReadImage(str(input_path)), str(output_path))

if __name__=="__main__":
    import sys
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    convert(input_path, output_path)