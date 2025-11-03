# Utility functions for I/O operations in the ablation module.
from __future__ import annotations
import os, yaml

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)