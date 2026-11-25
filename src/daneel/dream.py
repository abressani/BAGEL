"""Dream up new galaxies: a DCGAN trained on GalaxyMNIST.

A generator turns random noise vectors into synthetic 64x64 RGB galaxy-like
images, while a discriminator learns to tell them from the real GalaxyMNIST
images. The two networks are trained against each other (adversarially).

To run it from the terminal:
    daneel -i params_path.yaml --dream

All the settings (data and output paths, hyperparameters) are read from the
``dream`` section of the YAML file. The command saves a grid of generated
galaxies, a grid of real galaxies for comparison and a text report.

Reusable from Python:
    from daneel.dream import GalaxyGAN
    gan = GalaxyGAN({"data_root": "/abs/path/to/data", "epochs": 10})
    gan.train(gan.load_images())
    new_galaxies = gan.generate(16)   # (16, 64, 64, 3) floats in [0, 1]
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids OMP Error #15 on some macOS conda setups

import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from daneel.galaxies.morphology import IMAGE_SIZE, azimuthal_mean_profile

REAL_LABEL = 0.9   # one-sided label smoothing for the discriminator

DEFAULTS = {
    "split": "train",
    "epochs": 50,
    "batch_size": 64,
    "latent_dim": 100,
    "base_channels": 64,
    "lr": 0.0002,
    "beta1": 0.5,
    "seed": 84631,
    "grid_size": 8,
    "augment": True,
    "device": "cpu",
    "num_threads": None,
    "figure": "assignment2_taskI_gan.png",
    "real_figure": "assignment2_taskI_real.png",
    "report": "report_gan_assignment2_taskI.txt",
}


class Generator(nn.Module):
    """Noise vector (latent_dim) -> RGB image 3x64x64 with values in [-1, 1]."""

    def __init__(self, latent_dim=100, base_channels=64):
        super().__init__()
        c = base_channels

        def up(i, o, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

        self.net = nn.Sequential(
            *up(latent_dim, c * 8, 4, 1, 0),      # 1x1   -> 4x4
            *up(c * 8, c * 4),                    # 4x4   -> 8x8
            *up(c * 4, c * 2),                    # 8x8   -> 16x16
            *up(c * 2, c),                        # 16x16 -> 32x32
            nn.ConvTranspose2d(c, 3, 4, 2, 1),    # 32x32 -> 64x64
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """RGB image 3x64x64 -> one logit (large = judged real)."""

    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels

        def down(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),   # 64 -> 32
            *down(c, c * 2),                                                 # 32 -> 16
            *down(c * 2, c * 4),                                             # 16 -> 8
            *down(c * 4, c * 8),                                             # 8  -> 4
            nn.Conv2d(c * 8, 1, 4, 1, 0),                                    # 4  -> 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


class GalaxyGAN:
    """DCGAN that generates synthetic GalaxyMNIST-like galaxies.

    Parameters (keys of the ``dream`` section of the YAML file)
    ----------
    data_root : str or Path
        folder that contains ``GalaxyMNIST/raw/{train,test}_dataset.hdf5`` (required)
    output_dir : str or Path
        folder where the figures and the report are saved (default: current directory)
    split : str
        'train' (default) or 'test': the images the GAN learns from
    epochs, batch_size : int
        training length and mini-batch size (default 50 and 64)
    latent_dim : int
        length of the noise vector given to the generator (default 100)
    base_channels : int
        width of both networks (default 64; use 32 for a faster, smaller GAN)
    lr, beta1 : float
        Adam learning rate and first momentum (default 0.0002 and 0.5)
    seed : int
        random seed (weights, batch order and the noise of the final grid)
    grid_size : int
        the figure shows grid_size x grid_size galaxies (default 8)
    augment : bool
        random horizontal/vertical flips of the real images (default true)
    device : str
        'cpu' (default), 'cuda' or 'mps'
    num_threads : int
        CPU threads used by torch. By default torch's current setting is left
        unchanged. More than one thread can crash environments in which two
        OpenMP libraries are loaded (OMP Error #15 / #179, segmentation fault).
    figure, real_figure, report : str
        output file names
    """

    def __init__(self, params):
        if params.get("data_root") is None:
            raise ValueError("'dream' section is missing: data_root")
        s = {k: (params[k] if params.get(k) is not None else v) for k, v in DEFAULTS.items()}
        self.data_root = Path(params["data_root"]).expanduser()
        self.output_dir = Path(params.get("output_dir") or ".").expanduser()
        try:
            self.split = str(s["split"])
            self.epochs = int(s["epochs"])
            self.batch_size = int(s["batch_size"])
            self.latent_dim = int(s["latent_dim"])
            self.base_channels = int(s["base_channels"])
            self.lr = float(s["lr"])
            self.beta1 = float(s["beta1"])
            self.seed = int(s["seed"])
            self.grid_size = int(s["grid_size"])
            self.augment = bool(s["augment"])
            self.device = torch.device(str(s["device"]))
            self.num_threads = int(s["num_threads"]) if s["num_threads"] is not None else None
            self.figure = str(s["figure"])
            self.real_figure = str(s["real_figure"])
            self.report = str(s["report"])
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("invalid value in the 'dream' section (check epochs, lr, device, ...)") from None
        if self.split not in ("train", "test"):
            raise ValueError("dream.split must be 'train' or 'test'")
        if min(self.epochs, self.batch_size, self.latent_dim, self.base_channels, self.grid_size) < 1:
            raise ValueError("dream.epochs, batch_size, latent_dim, base_channels and grid_size must be positive")
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("dream.num_threads must be positive")
        if self.lr <= 0 or not 0.0 <= self.beta1 < 1.0:
            raise ValueError("dream.lr must be positive and dream.beta1 in [0, 1)")
        self.generator = None
        self.discriminator = None
        self.history = []        # per epoch: D loss, G loss, D(real), D(fake)
        self.train_seconds = 0.0

    #data
    def load_images(self):
        """Return the real images as a uint8 array (N, 64, 64, 3)"""
        path = self.data_root / "GalaxyMNIST" / "raw" / f"{self.split}_dataset.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        with h5py.File(path, "r") as f:
            images = np.asarray(f["images"])
        if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
            images = np.transpose(images, (0, 2, 3, 1))  # NCHW -> NHWC
        if images.shape[1:] != (*IMAGE_SIZE, 3):
            raise ValueError(f"expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB images, got shape {images.shape}")
        return images

    @staticmethod
    def _random_flips(x):
        """Flip every image horizontally and/or vertically with probability 1/2"""
        h = torch.rand(x.size(0), device=x.device) < 0.5
        v = torch.rand(x.size(0), device=x.device) < 0.5
        x = torch.where(h.view(-1, 1, 1, 1), x.flip(3), x)
        return torch.where(v.view(-1, 1, 1, 1), x.flip(2), x)

    #training
    def train(self, images):
        """Adversarial training; stores the loss history of every epoch"""
        torch.manual_seed(self.seed)
        x = torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()
        loader = DataLoader(
            TensorDataset(x), batch_size=self.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        if len(loader) == 0:
            raise ValueError("dream.batch_size is larger than the number of images")

        dev = self.device
        if str(dev).split(":")[0] == "cpu":
            if self.num_threads is not None:
                torch.set_num_threads(self.num_threads)
            print(f"Training on the CPU with {torch.get_num_threads()} thread(s)")
        self.generator = Generator(self.latent_dim, self.base_channels).to(dev)
        self.discriminator = Discriminator(self.base_channels).to(dev)
        self.generator.apply(_init_weights)
        self.discriminator.apply(_init_weights)
        loss_fn = nn.BCEWithLogitsLoss()
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.history = []
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            sums, n = np.zeros(4), 0
            self.generator.train()
            self.discriminator.train()
            for (real,) in loader:
                real = real.to(dev)
                b = real.size(0)
                if self.augment:
                    real = self._random_flips(real)
                fake = self.generator(torch.randn(b, self.latent_dim, device=dev))

                # discriminator: real -> REAL_LABEL (smoothed), generated -> 0
                opt_d.zero_grad()
                out_real = self.discriminator(real)
                out_fake = self.discriminator(fake.detach())
                loss_d = (loss_fn(out_real, torch.full((b,), REAL_LABEL, device=dev))
                          + loss_fn(out_fake, torch.zeros(b, device=dev)))
                loss_d.backward()
                opt_d.step()

                # generator: make the discriminator call the generated images real
                opt_g.zero_grad()
                loss_g = loss_fn(self.discriminator(fake), torch.ones(b, device=dev))
                loss_g.backward()
                opt_g.step()

                sums += [loss_d.item(), loss_g.item(),
                         torch.sigmoid(out_real).mean().item(), torch.sigmoid(out_fake).mean().item()]
                n += 1
            d_loss, g_loss, d_real, d_fake = sums / n
            self.history.append(dict(d_loss=d_loss, g_loss=g_loss, d_real=d_real, d_fake=d_fake))
            elapsed = time.time() - start
            left = elapsed / epoch * (self.epochs - epoch)
            print(f"Epoch {epoch}/{self.epochs}: D loss={d_loss:.3f}; G loss={g_loss:.3f}; "
                  f"D(real)={d_real:.2f}; D(fake)={d_fake:.2f} "
                  f"[{elapsed / 60:.1f} min elapsed, ~{left / 60:.0f} min left]")
        self.train_seconds = time.time() - start

    def generate(self, n, seed=None):
        """Return n synthetic galaxies as floats in [0, 1], shape (n, 64, 64, 3)"""
        if self.generator is None:
            raise RuntimeError("Call train() or run() before generate()")
        gen = torch.Generator().manual_seed(self.seed if seed is None else seed)
        z = torch.randn(n, self.latent_dim, generator=gen).to(self.device)
        self.generator.eval()
        with torch.no_grad():
            out = self.generator(z).cpu()
        return ((out + 1.0) / 2.0).clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()

    #outputs
    @staticmethod
    def make_grid(images, nrow, pad=2):
        """Tile (N, H, W, 3) images into one nrow x nrow picture (white border)"""
        _, h, w, c = images.shape
        grid = np.ones((nrow * (h + pad) + pad, nrow * (w + pad) + pad, c), dtype=np.float32)
        for k in range(min(len(images), nrow * nrow)):
            r, col = divmod(k, nrow)
            y, x = pad + r * (h + pad), pad + col * (w + pad)
            grid[y:y + h, x:x + w] = images[k]
        return grid

    def save_grid(self, images, title, path):
        fig, ax = plt.subplots(figsize=(7, 7.3))
        ax.imshow(self.make_grid(images, self.grid_size), interpolation="nearest")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def diagnostics(self, real_images, n=256):
        """Crude comparison of real and generated galaxies (no realism metric)"""
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(real_images), size=min(n, len(real_images)), replace=False)
        real = real_images[idx].astype(np.float32) / 255.0
        fake = self.generate(len(idx), seed=self.seed + 1)

        def spread(a):  # mean pairwise pixel distance: low = images look alike
            flat = a.reshape(len(a), -1).astype(np.float64)
            sq = (flat ** 2).sum(1)
            d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0)
            return float(np.sqrt(d2[np.triu_indices(len(a), k=1)]).mean())

        def profile(a):
            return np.mean([azimuthal_mean_profile(im)[1] for im in a], axis=0)

        return {
            "n": len(idx),
            "rgb_real": real.mean(axis=(0, 1, 2)), "rgb_fake": fake.mean(axis=(0, 1, 2)),
            "spread_real": spread(real), "spread_fake": spread(fake),
            "profile_real": profile(real), "profile_fake": profile(fake),
        }

    def save_report(self, path, n_images, stats):
        h = self.history
        last = h[-min(5, len(h)):]
        mean_last = {k: float(np.mean([e[k] for e in last])) for k in h[0]}
        radii = [r for r in (0, 8, 16, 24, 31) if r < len(stats["profile_real"])]
        lines = [
            "GAN ON GALAXYMNIST (assignment 2, task I)",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "TRAINING SETUP",
            f"  Data: {n_images} real GalaxyMNIST images ({self.split} split), 64x64 RGB, scaled to [-1, 1].",
            f"  Augmentation: random horizontal and vertical flips = {self.augment}. No class labels are used.",
            f"  Generator (DCGAN, {_n_params(self.generator)} parameters): noise vector of length {self.latent_dim} ->",
            f"    4 transposed-convolution blocks (4x4, 8x8, 16x16, 32x32; BatchNorm + ReLU) -> 64x64 RGB (Tanh).",
            f"  Discriminator ({_n_params(self.discriminator)} parameters): 4 strided convolutions (64 -> 4 pixels; "
            "LeakyReLU 0.2, BatchNorm) -> 1 logit.",
            f"  Base width {self.base_channels} channels in both networks; weights initialised N(0, 0.02).",
            f"  Loss: binary cross-entropy. The discriminator sees real images with label {REAL_LABEL} (one-sided "
            "label smoothing) and generated images with label 0;",
            "    the generator is trained to make the discriminator output 1 for its images (non-saturating loss).",
            f"  Optimiser: Adam, lr={self.lr}, betas=({self.beta1}, 0.999) for both networks; one update of each "
            "network per mini-batch.",
            f"  {self.epochs} epochs, batch size {self.batch_size}, seed {self.seed}, device {self.device}; "
            f"training took {self.train_seconds / 60:.1f} min.",
            f"  Epoch 1: D loss={h[0]['d_loss']:.3f}, G loss={h[0]['g_loss']:.3f}, D(real)={h[0]['d_real']:.2f}, "
            f"D(fake)={h[0]['d_fake']:.2f}",
            f"  Last {len(last)} epochs (mean): D loss={mean_last['d_loss']:.3f}, G loss={mean_last['g_loss']:.3f}, "
            f"D(real)={mean_last['d_real']:.2f}, D(fake)={mean_last['d_fake']:.2f}",
            f"  Figures: {self.figure} (generated galaxies, fixed noise from the seed), {self.real_figure} "
            "(real galaxies for comparison).",
            "",
            f"CRUDE COMPARISON OF {stats['n']} REAL AND {stats['n']} GENERATED IMAGES",
            "  Mean R, G, B: real " + ", ".join(f"{v:.3f}" for v in stats["rgb_real"])
            + " | generated " + ", ".join(f"{v:.3f}" for v in stats["rgb_fake"]),
            f"  Mean pairwise pixel distance (diversity): real {stats['spread_real']:.2f}, "
            f"generated {stats['spread_fake']:.2f}",
            "  Mean radial intensity profile at radius " + ", ".join(str(r) for r in radii) + " pixels:",
            "    real      " + ", ".join(f"{stats['profile_real'][r]:.3f}" for r in radii),
            "    generated " + ", ".join(f"{stats['profile_fake'][r]:.3f}" for r in radii),
            "",
            "LIMITATIONS OF THE SYNTHETIC IMAGES",
            "  - Small, low-resolution training set: 64x64 images (downsampled from 224x224) of a few thousand galaxies,",
            "    so fine structure such as faint spiral arms, bars and compact companions is lost or invented, and",
            "    generated images can show blur, checkerboard artefacts or blobs that are not galaxies.",
            "  - Unconditional model: the classes (round, cigar, edge-on, spiral) cannot be requested, and the class",
            "    mix of the generated images is unknown and may differ from the real one.",
            "  - GAN training is unstable and the losses are not a quality measure. Mode collapse (many similar",
            "    images) or memorisation of training images are possible; the diversity number above is only a",
            "    crude check. No realism metric such as FID is computed; quality is judged by eye.",
            "  - Not physical: the images are RGB composites without calibrated photometry, noise or point-spread",
            "    function, so generated galaxies have no measured sizes, colours or masses and must not be used",
            "    for scientific measurements.",
            "  - Single training run and seed; another seed or more epochs can give different samples.",
        ]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    #run
    def run(self):
        """Full pipeline: load, train, generate, save figures and report."""
        images = self.load_images()
        print(f"Training set: {len(images)} galaxies ({self.split} split); "
              f"{self.epochs} epochs on {self.device}")
        self.train(images)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        n_show = self.grid_size ** 2
        fake = self.generate(n_show)
        figure = self.output_dir / self.figure
        self.save_grid(fake, "Synthetic galaxies generated by the GAN", figure)

        rng = np.random.default_rng(self.seed)
        real_idx = rng.choice(len(images), size=min(n_show, len(images)), replace=False)
        real_figure = self.output_dir / self.real_figure
        self.save_grid(images[real_idx].astype(np.float32) / 255.0, "Real GalaxyMNIST galaxies", real_figure)

        report = self.output_dir / self.report
        self.save_report(report, len(images), self.diagnostics(images))
        print(f"Saved: {figure.resolve()}")
        print(f"Saved: {real_figure.resolve()}")
        print(f"Saved: {report.resolve()}")
        return fake
