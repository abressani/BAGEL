"""Reusable Random Forest morphology-classifier for GalaxyMNIST.
The class reads the images, computes the features (radial brightness
profile and mean colour), trains a Random Forest for each configuration
given in the YAML file, picks the best one on the validation set and
tests it on the test set.

To run it from the terminal:
    daneel -i params_path.yaml -d rf
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed (e.g. inside Docker)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from daneel.galaxies.morphology import CLASS_NAMES, azimuthal_mean_profile

DEFAULT_HYPERPARAMETERS = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]


class GalaxyRandomForest:
    """Random Forest classifier of 4 galaxy morphology classes.

    Parameters
    ----------
    data_root : str or Path
        folder that contains ``GalaxyMNIST/raw/{train,test}_dataset.hdf5``
    output_dir : str or Path
        folder where report, metrics and figure are saved
    seed : int
        random seed (train/validation split and forests)
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

    def __init__(
        self,
        data_root,
        output_dir=".",
        seed=10,
        n_profile_bins=32,
        use_colour=True,
        validation_size=0.2,
        hyperparameters=None,
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.n_profile_bins = int(n_profile_bins)
        self.use_colour = bool(use_colour)
        self.validation_size = float(validation_size)
        self.hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
        self.results = []
        self.best = None
        self.test_metrics = None

    #setup
    def __init__(self, params):
        features = params.get("features", {})
        self.data_root = Path(params["data_root"])
        self.output_dir = Path(params.get("output_dir", "."))
        self.seed = params.get("seed", 10)
        self.n_profile_bins = features.get("n_profile_bins", 32)
        self.use_colour = features.get("use_colour", True)
        self.validation_size = params.get("validation_size", 0.2)
        self.hyperparameters = params.get("hyperparameters", DEFAULT_HYPERPARAMETERS)
        self.results = []
        self.best = None
        self.test_metrics = None

    #data
    def load_split(self, split):
        """Return images (N, 64, 64, 3) and labels (N,) of 'train' or 'test'"""
        path = self.data_root / "GalaxyMNIST" / "raw" / f"{split}_dataset.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        with h5py.File(path, "r") as f:
            return f["images"][:], f["labels"][:].astype(int)

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
    def save_confusion_matrix(self, cm, title, path):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

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