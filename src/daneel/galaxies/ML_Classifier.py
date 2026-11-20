"""Machine-learning classifiers for the four GalaxyMNIST morphology classes.

Two classifiers share the same data loading, evaluation and confusion-matrix
code (``GalaxyClassifier``):

* ``GalaxyRandomForest``: Random Forest on radial-profile (+ mean colour)
  features. It trains one forest per configuration given in the YAML file,
  picks the best one on the validation set and tests it on the test set.
* ``GalaxyCNN``: PyTorch convolutional neural network trained on the images.

To run them from the terminal:
    daneel -i params_path.yaml -d rf
    daneel -i params_path.yaml -d cnn

PyTorch is imported only when ``GalaxyCNN`` is used, so the Random Forest and
the other daneel commands do not need torch.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed (e.g. inside Docker)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from daneel.galaxies.morphology import CLASS_NAMES, IMAGE_SIZE, azimuthal_mean_profile

DEFAULT_HYPERPARAMETERS = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]


def _format_confusion(cm):
    """Confusion matrix as text, with the class names as row/column labels."""
    names = list(CLASS_NAMES)
    width = max(len(n) for n in names) + 2
    head = " " * width + "".join(f"{n:>17}" for n in names)
    rows = [f"{names[i]:<{width}}" + "".join(f"{int(v):>17d}" for v in cm[i]) for i in range(len(names))]
    return "(rows = true class, columns = predicted class)\n" + "\n".join([head] + rows)


def _per_class(cm):
    """Per-class precision and recall from a confusion matrix."""
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return "\n".join(
        f"  {CLASS_NAMES[i]:<17} precision={precision[i]:.3f}  recall={recall[i]:.3f}"
        for i in range(len(CLASS_NAMES))
    )


class GalaxyClassifier:
    """Shared base: parameters, data loading, evaluation and plotting.

    Parameters (keys of the YAML section)
    ----------
    data_root : str or Path
        folder that contains ``GalaxyMNIST/raw/{train,test}_dataset.hdf5``
    output_dir : str or Path
        folder where report, metrics and figures are saved
    seed : int
        random seed (train/validation split and model)
    """

    default_seed = 10

    def __init__(self, params):
        self.data_root = Path(params["data_root"]).expanduser()
        self.output_dir = Path(params.get("output_dir", ".")).expanduser()
        self.seed = params.get("seed", self.default_seed)
        self.test_metrics = None

    #data
    def load_split(self, split):
        """Return images (N, 64, 64, 3) and labels (N,) of 'train' or 'test'"""
        path = self.data_root / "GalaxyMNIST" / "raw" / f"{split}_dataset.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        with h5py.File(path, "r") as f:
            return f["images"][:], f["labels"][:].astype(int)

    #evaluation
    def evaluate(self, y_true, y_pred):
        """Confusion matrix, accuracy, macro precision and macro recall"""
        return {
            "cm": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    #outputs
    def save_confusion_matrix(self, cm, title, path, figsize=(5.5, 4.5), dpi=150):
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


class GalaxyRandomForest(GalaxyClassifier):
    """Random Forest classifier of 4 galaxy morphology classes.

    Extra parameters (besides those of ``GalaxyClassifier``)
    ----------
    n_profile_bins : int
        number of radial-profile bins used as features (max 32 for 64x64 images)
    use_colour : bool
        if True, append the mean R, G, B colour of each image to the features
    validation_size : float
        fraction of the official train set used for validation (stratified)
    hyperparameters : list of dict
        each dict has ``n_estimators`` and ``max_depth`` (null = unlimited)
    The configuration with the best validation accuracy is selected.
    """

    #setup
    def __init__(self, params):
        super().__init__(params)
        features = params.get("features", {})
        self.n_profile_bins = features.get("n_profile_bins", 32)
        self.use_colour = features.get("use_colour", True)
        self.validation_size = params.get("validation_size", 0.2)
        self.hyperparameters = params.get("hyperparameters", DEFAULT_HYPERPARAMETERS)
        self.results = []
        self.best = None

    #data
    def build_features(self, images):
        """Radial brightness profile (+ mean RGB colour if enabled) per image"""
        profiles = np.array(
            [azimuthal_mean_profile(img)[1][: self.n_profile_bins] for img in images]
        )
        if not self.use_colour:
            return profiles
        colour = (images.astype(float) / 255.0).mean(axis=(1, 2))
        return np.concatenate([profiles, colour], axis=1)

    def feature_names(self):
        names = [f"profile bin {i}" for i in range(self.n_profile_bins)]
        return names + ["mean R", "mean G", "mean B"] if self.use_colour else names

    #training
    def fit(self, X_train, y_train, X_val, y_val):
        """Train one forest per configuration, evaluate on validation data"""
        self.results = []
        for hp in self.hyperparameters:
            model = RandomForestClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp.get("max_depth"),
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            metrics = self.evaluate(y_val, model.predict(X_val))
            metrics.update(
                n_estimators=hp["n_estimators"],
                max_depth=hp.get("max_depth"),
                model=model,
            )
            self.results.append(metrics)
        self.best = max(self.results, key=lambda r: (r["accuracy"], r["precision"]))
        return self.best

    def predict(self, images):
        """Predict class indices for new images using the selected model"""
        if self.best is None:
            raise RuntimeError("Call fit() or run() before predict()")
        return self.best["model"].predict(self.build_features(images))

    #outputs
    def save_report(self, path, n_train, n_val, n_test):
        names = self.feature_names()
        lines = [
            "Random Forest on GalaxyMNIST",
            "=" * 60,
            f"Random seed: {self.seed}",
            f"Classes (rows/columns): {', '.join(CLASS_NAMES)}",
            f"Fitting set: {n_train}, validation set: {n_val}, test set: {n_test}",
            f"Features: {len(names)} (profile bins: {self.n_profile_bins}, "
            f"mean colour: {self.use_colour})",
            "",
        ]
        for k, r in enumerate(self.results, start=1):
            lines += [
                f"Configuration {k}: n_estimators={r['n_estimators']}, "
                f"max_depth={r['max_depth']}",
                "Validation confusion matrix (rows = true, columns = predicted):",
                str(r["cm"]),
                f"Validation accuracy: {r['accuracy']:.4f}",
                f"Validation macro precision: {r['precision']:.4f}",
                f"Validation macro recall: {r['recall']:.4f}",
                "",
            ]
        t = self.test_metrics
        lines += [
            f"Selected: n_estimators={self.best['n_estimators']}, "
            f"max_depth={self.best['max_depth']} (best validation accuracy)",
            "",
            "Official test set (evaluated once):",
            str(t["cm"]),
            f"Test accuracy: {t['accuracy']:.4f}",
            f"Test macro precision: {t['precision']:.4f}",
            f"Test macro recall: {t['recall']:.4f}"
            ""
        ]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    #run
    def run(self):
        """Full pipeline: load, featurise, validate, test, save outputs."""
        images_train, y_train_all = self.load_split("train")
        images_test, y_test = self.load_split("test")
        print(f"Train: {len(y_train_all)} galaxies, test: {len(y_test)} galaxies")

        X_train_all = self.build_features(images_train)
        X_test = self.build_features(images_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all,
            y_train_all,
            test_size=self.validation_size,
            stratify=y_train_all,
            random_state=self.seed,
        )

        self.fit(X_train, y_train, X_val, y_val)
        for r in self.results:
            print(
                f"n_estimators={r['n_estimators']}, max_depth={r['max_depth']}: "
                f"accuracy={r['accuracy']:.4f}, precision={r['precision']:.4f}"
            )

        self.test_metrics = self.evaluate(y_test, self.best["model"].predict(X_test))
        print(
            f"Test accuracy: {self.test_metrics['accuracy']:.4f}, "
            f"precision: {self.test_metrics['precision']:.4f}"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
        report = self.output_dir / "report_random_forest.txt"
        figure = self.output_dir / "rf_test_confusion_matrix.png"
        self.save_report(report, n_train, n_val, n_test)
        self.save_confusion_matrix(
            self.test_metrics["cm"], "Random Forest - test set", figure)
        print(f"Saved: {report.resolve()}")
        print(f"Saved: {figure.resolve()}")
        return self.test_metrics


class GalaxyCNN(GalaxyClassifier):
    """Four-class PyTorch CNN trained directly on the 64x64 RGB images.

    The network has one block (3x3 convolution, ReLU, 2x2 max pooling) per entry
    of ``channels``, then global average pooling, dropout and a linear layer with
    four outputs. It is trained with Adam and cross-entropy loss, and the metrics
    of the final epoch are reported (no best-epoch selection).

    Extra parameters (besides those of ``GalaxyClassifier``)
    ----------
    channels : list of int
        number of filters of each convolutional block (default [16, 32, 64])
    dropout : float
        dropout probability before the output layer (default 0.0)
    lr : float
        Adam learning rate (default 0.001)
    epochs : int
        number of training epochs (default 60)
    batch_size : int
        mini-batch size (default 32)
    validation_fraction : float
        fraction of the official train set used for validation (stratified)
    evaluate_test : bool
        also evaluate on the official test set. Use it only for the final,
        fixed settings: the test set must not be used to choose a model.

    PyTorch is imported inside the methods, so the rest of daneel does not need it.
    """

    default_seed = 84631
    DEFAULT_SETTINGS = {
        "channels": [16, 32, 64],
        "dropout": 0.0,
        "lr": 0.001,
        "epochs": 60,
        "batch_size": 32,
        "validation_fraction": 0.25,
        "evaluate_test": False,
    }

    #setup
    def __init__(self, params):
        super().__init__(params)
        s = {k: (params[k] if params.get(k) is not None else v)
             for k, v in self.DEFAULT_SETTINGS.items()}
        try:
            if not isinstance(s["channels"], (list, tuple)) or not s["channels"]:
                raise ValueError
            self.channels = tuple(int(c) for c in s["channels"])
            self.dropout = float(s["dropout"])
            self.lr = float(s["lr"])
            self.epochs = int(s["epochs"])
            self.batch_size = int(s["batch_size"])
            self.validation_fraction = float(s["validation_fraction"])
            self.evaluate_test = bool(s["evaluate_test"])
            self.seed = int(self.seed)
        except (TypeError, ValueError):
            raise ValueError(
                "invalid value in the 'cnn' section (check channels, dropout, lr, epochs, ...)"
            ) from None
        if min(self.channels) < 1:
            raise ValueError("cnn.channels must be positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("cnn.dropout must be in [0, 1)")
        if self.lr <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("cnn.lr, cnn.epochs and cnn.batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("cnn.validation_fraction must be between 0 and 1")
        self.model = None
        self.history = []   # (training loss, validation accuracy) per epoch
        self.results = {}

    #data
    def load_tensors(self, split):
        """Return float tensors (N, 3, 64, 64) in [0, 1] and int64 labels (N,)"""
        import torch

        images, labels = self.load_split(split)
        if images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
            images = np.transpose(images, (0, 2, 3, 1))  # NCHW -> NHWC
        if images.shape[1:] != (*IMAGE_SIZE, 3):
            raise ValueError(
                f"expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB images, got shape {images.shape}"
            )
        if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
            raise ValueError(f"labels must be in 0..{len(CLASS_NAMES) - 1}")
        x = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        return x, torch.from_numpy(labels.astype(np.int64))

    #model
    def build_model(self):
        """Seeded network: same initial weights for the same seed and settings"""
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        layers, in_ch = [], 3
        for out_ch in self.channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        return nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(in_ch, len(CLASS_NAMES)),
        )

    def _predict_loader(self, model, loader):
        import torch

        model.eval()
        true, pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                pred.extend(model(xb).argmax(1).numpy())
                true.extend(yb.numpy())
        return np.asarray(true), np.asarray(pred)

    #training
    def fit(self, fit_loader, val_loader):
        """Train for ``epochs`` epochs; record training loss and validation accuracy"""
        import torch
        from torch import nn

        model = self.build_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.history = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for xb, yb in fit_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)
            true, pred = self._predict_loader(model, val_loader)
            self.history.append((total / count, float((true == pred).mean())))
            print(f"Epoch {epoch}/{self.epochs}: training loss={self.history[-1][0]:.3f}; "
                  f"validation accuracy={self.history[-1][1]:.3f}")
        self.model = model
        return model

    def predict(self, images):
        """Predict class indices for images (N, 64, 64, 3) uint8"""
        if self.model is None:
            raise RuntimeError("Call fit() or run() before predict()")
        import torch

        x = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0)
        self.model.eval()
        with torch.no_grad():
            return self.model(x.permute(0, 3, 1, 2).contiguous()).argmax(1).numpy()

    #outputs
    def save_report(self, path, n_fit, n_val, n_test):
        lines = [
            "CNN on GalaxyMNIST",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Classes (rows/columns): {', '.join(CLASS_NAMES)}",
            f"channels={self.channels}, dropout={self.dropout}, lr={self.lr}, "
            f"epochs={self.epochs}, batch_size={self.batch_size}, seed={self.seed}",
            "Optimiser: Adam; loss: CrossEntropyLoss; pixels divided by 255; no augmentation",
            f"Fitting set: {n_fit}, validation set: {n_val}"
            + (f", test set: {n_test}" if n_test else ""),
            f"Final training loss: {self.history[-1][0]:.4f}",
            "",
        ]
        for name, m in self.results.items():
            title = "Validation set" if name == "validation" else "Official test set"
            lines += [
                f"{title} (final epoch):",
                _format_confusion(m["cm"]),
                f"Accuracy: {m['accuracy']:.4f}",
                f"Macro precision: {m['precision']:.4f}",
                f"Macro recall: {m['recall']:.4f}",
                "Per-class precision and recall:",
                _per_class(m["cm"]),
                "",
            ]
        if "test" not in self.results:
            lines.append("Official test set: not evaluated (set evaluate_test: true).")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    #run
    def run(self):
        """Full pipeline: load, split, train, evaluate, save outputs."""
        from torch.utils.data import DataLoader, TensorDataset

        x, y = self.load_tensors("train")
        fit_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.validation_fraction,
            random_state=self.seed,
            stratify=y.numpy(),
        )
        fit_loader = DataLoader(
            TensorDataset(x[fit_idx], y[fit_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            TensorDataset(x[val_idx], y[val_idx]), batch_size=self.batch_size, shuffle=False
        )
        print(f"Fit: {len(fit_idx)} galaxies, validation: {len(val_idx)} galaxies; "
              f"channels={self.channels}, lr={self.lr}, epochs={self.epochs}")

        self.fit(fit_loader, val_loader)

        self.results = {"validation": self.evaluate(*self._predict_loader(self.model, val_loader))}
        n_test = None
        if self.evaluate_test:
            x_test, y_test = self.load_tensors("test")
            test_loader = DataLoader(
                TensorDataset(x_test, y_test), batch_size=self.batch_size, shuffle=False
            )
            self.results["test"] = self.evaluate(*self._predict_loader(self.model, test_loader))
            self.test_metrics = self.results["test"]
            n_test = len(y_test)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, m in self.results.items():
            print(f"\n{name.upper()}: accuracy={m['accuracy']:.4f}, "
                  f"precision={m['precision']:.4f}, recall={m['recall']:.4f}")
            print(_format_confusion(m["cm"]))
            figure = self.output_dir / f"cnn_confusion_matrix_{name}.png"
            self.save_confusion_matrix(m["cm"], f"CNN - {name} set", figure, figsize=(7, 6), dpi=200)

        report = self.output_dir / "cnn_report.txt"
        weights = self.output_dir / "cnn_weights.pt"
        self.save_report(report, len(fit_idx), len(val_idx), n_test)
        torch.save(self.model.state_dict(), weights)
        print(f"\nSaved: {report.resolve()}")
        print(f"Saved: {weights.resolve()}")
        return self.results
