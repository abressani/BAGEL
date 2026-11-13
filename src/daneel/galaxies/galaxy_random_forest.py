"""This script aims to classify the galaxies in the dataset GalaxyMNIST based on thei morophology claas 
and using a Random Forest classifier"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from daneel.galaxies.morphology import (CLASS_NAMES, HF_DATASET_ID, IMAGE_SIZE, azimuthal_mean_profile)

#seed so the random forest is reproducible
SEED = 10  #fav numb
# four different combinations to compare: (n_estimators, max_depth)
CONFIGS = [(50, 5), (100, 10), (200, 10), (200, None)]
# Only the first 32 radial bins are used, beyond the annulii are cut by the edges of the 64x64 image and are no longer complete rings.
N_PROFILE_BINS = 32

# loading the split dataset (andd error if file not found)
def load_split(data_root, split):
    """Return all images (N, 64, 64, 3) and labels (N,) of 'train' or 'test'."""
    path = Path(data_root) / "GalaxyMNIST" / "raw" / f"{split}_dataset.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        return f["images"][:], f["labels"][:].astype(int)

#extracting features from images and saving them in an array
def build_features(images):
    """One row of 35 numbers per image: radial profile and mean colour (RGB)."""
    a = images.astype(float) / 255.0
    colour = a.mean(axis=(1, 2))  # mean R, G, B of each image -> (N, 3)
    profiles = np.array(
        [azimuthal_mean_profile(img)[1][:N_PROFILE_BINS] for img in images]
    )  # (N, 32)
    return np.concatenate([profiles, colour], axis=1)

#evaluating the combinations
def evaluate(y_true, y_pred):
    """Confusion matrix, accuracy and macro-averaged precision."""
    return {
        "cm": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }

# trying the 4 configurations
def run_experiment(X_train, y_train, X_val, y_val):
    """Train one forest per configuration and evaluate it on validation data."""
    results = []
    for n_estimators, max_depth in CONFIGS:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=SEED
        )
        model.fit(X_train, y_train)
        metrics = evaluate(y_val, model.predict(X_val))
        metrics.update(n_estimators=n_estimators, max_depth=max_depth, model=model)
        results.append(metrics)
    return results

#plotting a confusion matrix
def save_confusion_matrix(cm, title, path):
    """Save a confusion-matrix figure (rows = true, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

#tried to write a report
def write_report(path, n_features, n_train, n_val, n_test, results, best, test, importance_lines):
    lines = [
        "Random Forest on GalaxyMNIST - Assignment 2, Task D",
        "=" * 60,
        f"Random seed: {SEED}",
        f"Classes (rows/columns of the matrices): {', '.join(CLASS_NAMES)}",
        f"Fitting set: {n_train} galaxies, validation set: {n_val} galaxies "
        "(20% of the official train set, stratified), "
        f"official test set: {n_test} galaxies",
        f"Number of features: {n_features}",
        "Features: radial brightness profile (first 32 bins) and mean R, G, B colour.",
        "The radial profile keeps how light is concentrated with radius; "
        "the mean colour recovers part of the colour information lost in the profile plot.",
        "",
    ]
    for k, r in enumerate(results, start=1):
        lines += [
            f"Configuration {k}: n_estimators={r['n_estimators']}, max_depth={r['max_depth']}",
            "Validation confusion matrix (rows = true, columns = predicted):",
            str(r["cm"]),
            f"Validation accuracy: {r['accuracy']:.4f}",
            f"Validation macro precision: {r['precision']:.4f}",
            "",
        ]
    lines += [
        f"Selected configuration: n_estimators={best['n_estimators']}, "
        f"max_depth={best['max_depth']}",
        "Selection rule: highest validation accuracy (ties broken by macro precision).",
        "The selected model is not retrained on train+validation before the test.",
        "",
        "Evaluation on the official test set (done once):",
        str(test["cm"]),
        f"Test accuracy: {test['accuracy']:.4f}",
        f"Test macro precision: {test['precision']:.4f}",
    ]
    lines += ["", "Feature importance of the selected model:"] + importance_lines
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
 
#running with main built in the py file
def main():
    parser = argparse.ArgumentParser(description="Random Forest on GalaxyMNIST (Task D).")
    parser.add_argument("--data-root", required=True, help="Dataset root (absolute path)")
    parser.add_argument("--output-dir", default=".", help="Folder for report and figure")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
 
    #train and test sets
    images_train, y_train_all = load_split(args.data_root, "train")
    images_test, y_test = load_split(args.data_root, "test")
    print(f"Train: {len(y_train_all)} galaxies, test: {len(y_test)} galaxies")
 
    #features
    X_train_all = build_features(images_train)
    X_test = build_features(images_test)
 
    #validation set, taken from the train set only
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, stratify=y_train_all, random_state=SEED
    )
 
    #four configurations compared on validation data
    results = run_experiment(X_train, y_train, X_val, y_val)
    for r in results:
        print(f"n_estimators={r['n_estimators']}, max_depth={r['max_depth']}: "
              f"accuracy={r['accuracy']:.4f}, precision={r['precision']:.4f}")
 
    #best configuration on validation, then ONE evaluation on the test set
    best = max(results, key=lambda r: (r["accuracy"], r["precision"]))
    test = evaluate(y_test, best["model"].predict(X_test))

    names = [f"profile bin {i}" for i in range(N_PROFILE_BINS)] + ["mean R", "mean G", "mean B"]
    imp = best["model"].feature_importances_
    top = np.argsort(imp)[::-1][:8]
    print(f"Test accuracy: {test['accuracy']:.4f}, precision: {test['precision']:.4f}")

    importance_lines = [
        f"Total importance, radial profile: {imp[:N_PROFILE_BINS].sum():.3f}",
        f"Total importance, mean colour: {imp[N_PROFILE_BINS:].sum():.3f}",
        "Most important features:",
    ] + [f"  {names[i]}: {imp[i]:.3f}" for i in top] 

    #report and figure
    report_path = out_dir / "report_random_forest_assignment2_taskD.txt"
    write_report(report_path, X_train_all.shape[1], len(y_train), len(y_val),
                 len(y_test), results, best, test, importance_lines)
    save_confusion_matrix(test["cm"], "Random Forest - test set",
                          out_dir / "rf_test_confusion_matrix.png")
    print(f"Saved: {report_path.resolve()}")
 
 
if __name__ == "__main__":
    main()