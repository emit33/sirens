# config.py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml
import torch
from dacite import from_dict, Config as DaciteConfig

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    dim_in: int = 2
    dim_hidden: int = 256
    dim_out: int = 1
    n_hidden_layers: int = 5
    w0: float = 30


@dataclass
class TrainingConfig:
    n_epochs: int = 100
    lr: float = 0.01
    resolution: int = 256
    grayscale: bool = True


@dataclass
class PathConfig:
    data_dir: Path = PROJECT_ROOT / "triangles"
    results_dir: Path = PROJECT_ROOT / "checkpoints"


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig

    @classmethod
    def from_yaml(cls, yaml_path=None):
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "config.yaml"

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Configure dacite for type conversions
        dacite_config = DaciteConfig(
            type_hooks={
                Path: lambda x: Path(x) if isinstance(x, str) else x,
                Tuple[int, ...]: lambda x: tuple(x) if isinstance(x, list) else x,
                torch.device: lambda s: torch.device(s) if isinstance(s, str) else s,
                float: lambda s: float(s) if isinstance(s, str) else s,
                int: lambda x: int(float(x)) if isinstance(x, (str, float)) else x,
            }
        )

        return from_dict(data_class=cls, data=config_dict, config=dacite_config)
