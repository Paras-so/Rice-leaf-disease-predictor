"""Download the four explicitly versioned torchvision checkpoints to this project."""

import os
from .data import ROOT

os.environ["TORCH_HOME"] = str(ROOT / "artifacts/cache/torch")

from .models import BUILDERS

if __name__ == "__main__":
    for name, (_, weights) in BUILDERS.items():
        print(f"Downloading/verifying {name} IMAGENET1K_V1", flush=True)
        state = weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        del state
