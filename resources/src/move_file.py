import shutil
from pathlib import Path
import os

def convert(input_path: Path, output_path: Path, name:str):
    input_path = list(input_path.glob('*.mha'))[0]
    output_path = output_path / (input_path.name.replace('.mha', f'{name}.mha'))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.symlink_to(input_path, target_is_directory=False)

if __name__=="__main__":
    import sys
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    name = str(sys.argv[3])
    convert(input_path, output_path, name)