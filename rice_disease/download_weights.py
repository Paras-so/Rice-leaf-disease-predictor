"""Download the four explicitly versioned torchvision checkpoints to this project."""

import os
import argparse
from pathlib import Path

if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "rice_disease"
from .data import ROOT

os.environ["TORCH_HOME"] = str(ROOT / "artifacts/cache/torch")

from .models import BUILDERS

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name, (_, weights) in BUILDERS.items():
        print(f"Downloading/verifying {name} IMAGENET1K_V1", flush=True)
        state = weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        del state
    print(f"All four checkpoints are available in {ROOT / 'artifacts/cache/torch/hub/checkpoints'}")
