from dataclasses import dataclass
from typing import Optional


@dataclass
class HimoConfig:
    root: str = "${env.raw_datasets_folder}/himo/unpacked-20261029/data"
    annotations: str = "./outputs/himo-small-stage-20261029/stage_candidates.json"
    prepared_data: str = "${env.datasets_folder}/himo_ordered.pth"
    contact_indices: str = "${env.datasets_folder}/himo_smplx_indices.npy"
    initialization_checkpoint: str = "${env.assets_folder}/gb_main.pth"
    conditions: Optional[str] = None
    condition_index: int = 0
