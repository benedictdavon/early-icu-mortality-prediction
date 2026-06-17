import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
base_path = Path(os.getenv("ICU_DATA_DIR", PROJECT_ROOT / "data")).resolve()

hosp_path = str(base_path / "hosp")
icu_path = str(base_path / "icu")
label_path = str(base_path / "label")
