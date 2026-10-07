"""Area arithmetic for an externally annotated mask; this is not a segmentation model."""

import numpy as np


def affected_area(mask):
    """Mask labels: 0 background, 1 unaffected leaf, 2 diseased leaf."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError("Provide a 2D label mask using only 0=background, 1=unaffected leaf, 2=diseased leaf.")
    leaf_pixels = int(np.count_nonzero(mask > 0))
    if not leaf_pixels:
        raise ValueError("The mask contains no leaf pixels; an area percentage cannot be calculated.")
    diseased_pixels = int(np.count_nonzero(mask == 2))
    return {"leaf_pixels": leaf_pixels, "diseased_pixels": diseased_pixels,
            "affected_area_percent": diseased_pixels / leaf_pixels * 100,
            "source": "User-supplied labelled mask, not model segmentation",
            "severity_grade": None}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, help="2D PNG/TIFF or .npy mask: 0=background, 1=unaffected, 2=diseased")
    args = parser.parse_args()
    if args.mask is None:
        parser.print_help()
    else:
        try:
            if args.mask.suffix.lower() == ".npy":
                mask = np.load(args.mask, allow_pickle=False)
            else:
                with Image.open(args.mask) as image:
                    mask = np.array(image)
            print(json.dumps(affected_area(mask), indent=2))
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Mask error: {exc}\n")
