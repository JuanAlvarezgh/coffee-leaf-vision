"""Verify that BRACOL and RoCoLe datasets are present and report instructions."""

import sys
from pathlib import Path

from loguru import logger

BRACOL_INSTRUCTIONS = """
BRACOL — Brazilian Arabica Coffee Leaf Dataset
---------------------------------------------
NOTE: The Mendeley zip is corrupted (missing ZIP central directory) and the old
Kaggle mirror was removed. Use the AUTHORS' Google Drive mirror instead — it is
the complete, valid dataset (~279 MB).

1. Download from the authors' Google Drive (linked from
   https://github.com/esgario/lara2018):
     https://drive.google.com/uc?id=15YHebAGrx1Vhv8-naave-R5o3Uo70jsm
   You can use gdown:
     pip install gdown
     gdown 15YHebAGrx1Vhv8-naave-R5o3Uo70jsm -O data/BRACOL_gdrive.zip
2. Unzip it. The archive contains coffee-datasets/{leaf,segmentation,symptom}.
3. Move the "symptom" folder to data/BRACOL/ so the structure is:
     data/BRACOL/train/{1_health,2_miner,3_rust,4_phoma,5_cercospora}/
     data/BRACOL/val/{...}/
     data/BRACOL/test/{...}/
   (The predefined train/val/test splits are merged automatically; we run our
    own stratified 5-fold cross-validation.)
"""

ROCOLE_INSTRUCTIONS = """
RoCoLe — Robusta Coffee Leaf Dataset
-----------------------------------
1. Go to: https://data.mendeley.com/datasets/c5yvn32dzg/2
2. Click "Download" (CC BY 4.0 license, no account needed).
3. Unzip the file under data/RoCoLe/
4. The folder should contain a CSV with annotations and an Images/ folder.
"""


def check_dataset(path: Path, name: str, instructions: str) -> bool:
    if not path.exists():
        logger.error(f"{name} not found at {path}")
        print(instructions)
        return False
    if not any(path.iterdir()):
        logger.error(f"{name} folder is empty at {path}")
        print(instructions)
        return False
    logger.info(f"{name} present at {path}")
    return True


def main() -> int:
    data_dir = Path("data")
    bracol = data_dir / "BRACOL"
    rocole = data_dir / "RoCoLe"

    bracol_ok = check_dataset(bracol, "BRACOL", BRACOL_INSTRUCTIONS)
    rocole_ok = check_dataset(rocole, "RoCoLe", ROCOLE_INSTRUCTIONS)

    if bracol_ok and rocole_ok:
        logger.info("All datasets present. You can run preprocess.py next.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
