"""GalaxyMNIST image and azimuthally averaged radial-profile utilities using 
the Hugging Face mirror GalaxyMNIST dataset that stores 224x224 RGB images. 
For the Assignment 1 workflow, images are resized to 64x64 to match the original 
dataset. 

"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = (
    "smooth_round", # flat fall-off
    "smooth_cigar", # steep fall-off
    "edge_on_disk", # steep fall-off
    "unbarred_spiral", 
)
HF_DATASET_ID = "matthieulel/galaxy_mnist"
IMAGE_SIZE = (64, 64)
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


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
            "Install them with 'python -m pip install datasets pillow'."
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
    index: int | list[int] = 10,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Plot one or more real GalaxyMNIST images and their radial brightness 
    profiles.

    `index` can be a single integer or a list of integers. The figure has the
    RGB images in the top row and in the bottom panel the radial
    brightness profiles together fr comparison, one colour per galaxy.

    Returns a dictionary with index, split, class id/name, output path and the
    list of per-galaxy data under "galaxies"
    """

    indices = [index] if isinstance(index, int) else list(index)
    if len(indices) == 0:
        raise ValueError("at least one galaxy is recquired")

    #load every galaxy 
    galaxies =[]
    for i in indices:
        image, label = _load_one_galaxy(data_root, split, i)
        if label not in range(len(CLASS_NAMES)):
            raise ValueError(f"Unexpected GalaxyMNIST label {label}; expected 0–3.")
        radii, profile = azimuthal_mean_profile(image)
        galaxies.append({"index": i, "class_id": label, "class_name": CLASS_NAMES[label], "image": image, "radii": radii,
                "profile": profile})
        
    if output_path is None:
        if len(indices) == 1:
            output_path = Path(f"galaxy_{indices[0]}.png")
        else:
            output_path = Path("galaxies_comparison_" + "_".join(map(str, indices)) + ".png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #plots for each galaxy

    n = len(galaxies)
    fig = plt.figure(figsize=(max(6.0, 2.6*n), 9.5))
    grid = fig.add_gridspec(2, n, height_ratios=[1, 1.2])

    first_profile_ax = None
    for k, g in enumerate(galaxies):
        color = COLORS[k % len(COLORS)]

        # images
        ax_image = fig.add_subplot(grid[0, k])
        ax_image.imshow(g["image"])
        ax_image.set_title(f"{g['class_name']}\n(index {g['index']})", color=color, fontsize=10)
        ax_image.set_xlabel("x [pixels]")
        if k == 0:
            ax_image.set_ylabel("y [pixels]")


    # profiles in same plot
    ax_profile = fig.add_subplot(grid[1, :])
    for k, g in enumerate(galaxies):
        ax_profile.plot(g["radii"], g["profile"], color=COLORS[k % len(COLORS)], marker="o", markersize=2.5, 
            linewidth=1.4, label=f"{g['class_name']} (index {g['index']})")
    
    ax_profile.set_xlabel("Radius [pixels]")
    ax_profile.set_ylabel("Mean relative intensity")
    ax_profile.set_title("Azimuthally averaged brightness")
    ax_profile.grid(alpha=0.3)
    ax_profile.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    #print statements
    if n == 1:
        result_index = galaxies[0]["index"]
        result_class_id = galaxies[0]["class_id"]
        result_class_name = galaxies[0]["class_name"]
    else:
        result_index = [g["index"] for g in galaxies]
        result_class_id = [g["class_id"] for g in galaxies]
        result_class_name = ", ".join(g["class_name"] for g in galaxies)

    return {
        "index": result_index,
        "split": split,
        "class_id": result_class_id,
        "class_name": result_class_name,
        "output_path": str(output_path.resolve()),
        "galaxies": galaxies,
        "image": galaxies[0]["image"],
        "radii": galaxies[0]["radii"],
        "profile": galaxies[0]["profile"],
    }

def count_galaxies_per_class(data_root: str | Path, split: str = "train"):
    """Return a dictionary {class_name: number of galaxies} for the given split"""
    hd5f_path = Path(data_root) / "GalaxyMNIST" / "raw" / f"{split}_dataset.hd5f"
    if hd5f_path.is_file():
        with h5py.File(hd5f_path, "r") as handle:
            labels = np.asarray(handle["labels"])
    else:
        from datasets import load_dataset
        labels = np.asarray(load_dataset(HF_DATASET_ID, split=split)["label"])

    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {name: int(c) for name, c in zip(CLASS_NAMES, counts)}


def main():
    parser = argparse.ArgumentParser(description="Plot real GalaxyMNIST galaxies and radial profiles.")
    parser.add_argument("--data-root", default="data", help="Dataset root (default: ./data)")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, nargs="+", default=[6401], help="Zero-based galaxy index (one or more)")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()
    result = plot_galaxy(data_root=args.data_root, split=args.split, index=args.index, output_path=args.output)

    print(f"DATA SOURCE: REAL GalaxyMNIST ({result['split']})")
    print(f"Galaxy index: {result['index']}")
    print(f"Morphological class: {result['class_name']}")
    print(f"Saved: {result['output_path']}")


if __name__ == "__main__":
    main()
