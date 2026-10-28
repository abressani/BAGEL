from pathlib import Path

import h5py
import numpy as np
from datasets import load_dataset
from PIL import Image

DATA_ROOT = Path("data")
raw = DATA_ROOT / "GalaxyMNIST" / "raw"
raw.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    ds = load_dataset("matthieulel/galaxy_mnist", split=split)
    images = np.empty((len(ds), 64, 64, 3), dtype=np.uint8)
    labels = np.empty(len(ds), dtype=np.int64)
    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB").resize(
            (64, 64), resample=Image.Resampling.LANCZOS
        )
        images[i] = np.asarray(img, dtype=np.uint8)
        labels[i] = sample["label"]
    with h5py.File(raw / f"{split}_dataset.hdf5", "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("labels", data=labels)
    print(split, images.shape, labels.shape)