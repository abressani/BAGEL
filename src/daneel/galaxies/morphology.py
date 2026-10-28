"""GalaxyMNIST image and azimuthally averaged radial-profile utilities.

The Hugging Face mirror stores 224x224 RGB images. For the Assignment 1
workflow, images are resized to 64x64 to match the original GalaxyMNIST
tutorial. Real data are required; this module never generates synthetic data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = (
    "smooth_round",
    "smooth_cigar",
    "edge_on_disk",
    "unbarred_spiral",
)
HF_DATASET_ID = "matthieulel/galaxy_mnist"
IMAGE_SIZE = (64, 64)


def _load_one_galaxy(data_root: str | Path, split: str, index: int):
    """Return one real RGB uint8 image (64x64) and its integer class label."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if index < 0:
        raise IndexError("index must be non-negative")

    # Prefer a local HDF5 copy if one exists.
    hdf5_path = Path(data_root) / "GalaxyMNIST" / "raw" / f"{split}_dataset.hdf5"
    if hdf5_path.is_file():
        with h5py.File(hdf5_path, "r") as handle:
            if index >= len(handle["images"]):
                raise IndexError(f"index must be between 0 and {len(handle['images']) - 1}")
            image = np.asarray(handle["images"][index])
            label = int(handle["labels"][index])
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[:2] != IMAGE_SIZE:
            from PIL import Image
            image = np.asarray(
                Image.fromarray(image.astype(np.uint8)).convert("RGB").resize(
                    IMAGE_SIZE, resample=Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
        return np.asarray(image, dtype=np.uint8), label

    # The official package's old Dropbox download links may no longer work.
    # The Hugging Face mirror reuses the cache populated by the notebook.
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Real GalaxyMNIST loading needs 'datasets' and 'Pillow'. "
            "Activate the ca26bagel environment and install them with "
            "`python -m pip install datasets pillow`."
        ) from exc

    dataset = load_dataset(HF_DATASET_ID, split=split)
    if index >= len(dataset):
        raise IndexError(f"index must be between 0 and {len(dataset) - 1} for split {split!r}")
    sample = dataset[index]
    image = sample["image"].convert("RGB").resize(
        IMAGE_SIZE, resample=Image.Resampling.LANCZOS
    )
    label = int(sample["label"])
    if label not in range(len(CLASS_NAMES)):
        raise ValueError(f"Unexpected GalaxyMNIST label {label}; expected 0–3.")
    return np.asarray(image, dtype=np.uint8), label


def azimuthal_mean_profile(image: np.ndarray):
    """Compute mean grayscale intensity in one-pixel-wide radial annuli.

    The image is centred on its geometric centre. Intensities are scaled to
    [0, 1]; this is a simple teaching-level profile, not calibrated photometry.
    Returns (radii_in_pixels, mean_relative_intensity).
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] < 3:
            raise ValueError("RGB image must have three colour channels.")
        array = array[..., :3].astype(float) / (255.0 if array.max() > 1 else 1.0)
        gray = (
            0.2126 * array[..., 0]
            + 0.7152 * array[..., 1]
            + 0.0722 * array[..., 2]
        )
    elif array.ndim == 2:
        gray = array.astype(float)
        if gray.max() > 1:
            gray = gray / 255.0
    else:
        raise ValueError("image must be a 2-D grayscale or 3-D RGB array.")

    yy, xx = np.indices(gray.shape)
    yc, xc = (np.array(gray.shape) - 1) / 2
    radius_pixel = np.hypot(xx - xc, yy - yc).astype(int)
    flux_sum = np.bincount(radius_pixel.ravel(), weights=gray.ravel())
    pixel_count = np.bincount(radius_pixel.ravel())
    valid = pixel_count > 0
    radii = np.arange(len(flux_sum))[valid]
    profile = flux_sum[valid] / pixel_count[valid]
    return radii, profile


def plot_galaxy(
    data_root: str | Path = "data",
    split: str = "train",
    index: int = 10,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Plot one real GalaxyMNIST image and its radial brightness profile.

    Returns a dictionary containing the index, class id/name, output path,
    image, radii and profile.
    """
    image, label = _load_one_galaxy(data_root, split, index)
    if label not in range(len(CLASS_NAMES)):
        raise ValueError(f"Unexpected GalaxyMNIST label {label}; expected 0–3.")
    radii, profile = azimuthal_mean_profile(image)

    if output_path is None:
        output_path = Path(f"galaxy_{index}_assignment1_taskF.png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_image, ax_profile) = plt.subplots(1, 2, figsize=(9.3, 4.0))
    ax_image.imshow(image)
    ax_image.set_title(f"Galaxy {index} — {CLASS_NAMES[label]}")
    ax_image.set_xlabel("x [pixels]")
    ax_image.set_ylabel("y [pixels]")

    ax_profile.plot(radii, profile, marker="o", markersize=2.5, linewidth=1.4)
    ax_profile.set_xlabel("Radius [pixels]")
    ax_profile.set_ylabel("Mean relative intensity")
    ax_profile.set_title("Azimuthally averaged brightness")
    ax_profile.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "index": index,
        "split": split,
        "class_id": label,
        "class_name": CLASS_NAMES[label],
        "output_path": str(output_path.resolve()),
        "image": image,
        "radii": radii,
        "profile": profile,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot a real GalaxyMNIST galaxy and radial profile.")
    parser.add_argument("--data-root", default="data", help="Dataset root (default: ./data)")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, default=10, help="Zero-based galaxy index")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()
    result = plot_galaxy(
        data_root=args.data_root,
        split=args.split,
        index=args.index,
        output_path=args.output,
    )
    print(f"DATA SOURCE: REAL GalaxyMNIST ({result['split']})")
    print(f"Galaxy index: {result['index']}")
    print(f"Morphological class: {result['class_name']} (label {result['class_id']})")
    print(f"Saved: {result['output_path']}")


if __name__ == "__main__":
    main()
