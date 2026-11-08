"""Centralized configuration for the Cat vs Dog classifier."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Project configuration object.
    
    Eliminates duplicated and conflicting settings across notebooks and scripts.
    """
    # Image specifications
    img_height: int = 150
    img_width: int = 150
    channels: int = 3

    # Training hyperparameters
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 0.001
    seed: int = 42

    # Directory paths (resolved relative to project root)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir_name: str = "cats_and_dogs"

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def validation_dir(self) -> Path:
        return self.data_dir / "validation"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def models_dir(self) -> Path:
        p = self.project_root / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_dir(self) -> Path:
        p = self.project_root / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Official freeCodeCamp 50-image test set answers (0 = cat, 1 = dog)
    fcc_test_answers: List[int] = field(
        default_factory=lambda: [
            1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
            0, 0, 0, 0, 0, 0
        ]
    )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        data["data_dir"] = str(self.data_dir)
        data["train_dir"] = str(self.train_dir)
        data["validation_dir"] = str(self.validation_dir)
        data["test_dir"] = str(self.test_dir)
        return data

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
