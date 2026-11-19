"""This script aims to classify the galaxies in the dataset GalaxyMNIST based on thei morophology class 
and using a Logisti Regression classifier"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from daneel.galaxies.morphology import CLASS_NAMES, azimuthal_mean_profile


#need scaling

#same parameters for random forest
SEED = 10
N_PROFILE_BINS = 32

#setting logistic regression param
C=1.0
#if we want to compare with random forest
RF_TEXT = "n_estimators=200, max_depth=None"
RF_VAL_ACCURACY = 0.6656 #both values to check with rf report
RF_VAL_PRECISION = 0.6686

#same functions of random forest for data loading, featres and evaluating
def load_split(data_root, split):
    """Return all images (N, 64, 64, 3) and labels (N,) of 'train' or 'test'."""
    path = Path(data_root) / "GalaxyMNIST" / "raw" / f"{split}_dataset.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        return f["images"][:], f["labels"][:].astype(int)

def build_features(images):
    """One row of 35 numbers per image: radial profile + mean R, G, B."""
    a = images.astype(float) / 255.0
    colour = a.mean(axis=(1, 2))  # mean R, G, B of each image -> (N, 3)
    profiles = np.array(
        [azimuthal_mean_profile(img)[1][:N_PROFILE_BINS] for img in images]
    )  # (N, 32)
    return np.concatenate([profiles, colour], axis=1)
 
def evaluate(y_true, y_pred):
    """Confusion matrix, accuracy and macro-averaged precision."""
    return {
        "cm": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }
 

def write_report(path, n_features, n_train, n_val, n_test, val, test):
    error = np.sqrt(val["accuracy"] * (1 - val["accuracy"]) / n_val)
    lines = [
        "Logistic Regression on GalaxyMNIST - Assignment 2, Task E",
        "=" * 60,
        f"Model: LogisticRegression(C={C}, max_iter=1000), multi-class (4 classes)",
        "Feature scaling: StandardScaler (mean 0, std 1 for every feature), fitted "
        "on the fitting set only, using a scikit-learn Pipeline.",
        f"Random seed: {SEED}",
        f"Classes (rows/columns of the matrices): {', '.join(CLASS_NAMES)}",
        f"Fitting set: {n_train} galaxies, validation set: {n_val} galaxies "
        "(20% of the official train set, stratified), "
        f"official test set: {n_test} galaxies",
        f"Number of features: {n_features}",
        "Features: radial brightness profile (first 32 bins) and mean R, G, B colour "
        "",
        "Validation confusion matrix (rows = true, columns = predicted):",
        str(val["cm"]),
        f"Validation accuracy: {val['accuracy']:.4f}",
        f"Validation macro precision: {val['precision']:.4f}",
        "",
        "Comparison with the Random Forest baseline (validation set):",
        f"  Random Forest ({RF_TEXT}): accuracy {RF_VAL_ACCURACY:.4f}, "
        f"macro precision {RF_VAL_PRECISION:.4f}",
        f"  Logistic Regression: accuracy {val['accuracy']:.4f}, "
        f"macro precision {val['precision']:.4f}",
        f"  Difference (Logistic Regression - Random Forest): "
        f"accuracy {val['accuracy'] - RF_VAL_ACCURACY:+.4f}, "
        f"macro precision {val['precision'] - RF_VAL_PRECISION:+.4f}",
        f"  Rough statistical error on one accuracy ({n_val} galaxies): "
        f"Approximate binomial standard error of validation accuracy: {error:.4f}",
        "",
        "Evaluation on the official test set (done once, on this final model):",
        str(test["cm"]),
        f"Test accuracy: {test['accuracy']:.4f}",
        f"Test macro precision: {test['precision']:.4f}",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
 
def run_logistic_regression(data_root, output_dir="."):
    """Run the logistic regression with same parameters of random forest."""
    out_dir = Path(output_dir)

    #train and test set 
    images_train, y_train_all = load_split(data_root, "train")
    images_test, y_test = load_split(data_root, "test")
    print(f"Train: {len(y_train_all)} galaxies, test: {len(y_test)} galaxies")
 
    #features
    X_train_all = build_features(images_train)
    X_test = build_features(images_test)
 
    #validation set, taken from the train set only
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, stratify=y_train_all, random_state=SEED
    )
 
    #scaler + classifier 
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    model.fit(X_train, y_train)
 
    val = evaluate(y_val, model.predict(X_val))
    print(f"Validation accuracy: {val['accuracy']:.4f}, precision: {val['precision']:.4f}")
 
    test = evaluate(y_test, model.predict(X_test))
    print(f"Test accuracy: {test['accuracy']:.4f}, precision: {test['precision']:.4f}")
 
    #report and figure
    report_path = out_dir / "report_logistic_regression_assignment2_taskE.txt"
    write_report(report_path, X_train_all.shape[1], len(y_train), len(y_val),
                 len(y_test), val, test)
    save_confusion_matrix(test["cm"], "Logistic Regression - test set",
                          out_dir / "lr_test_confusion_matrix.png")
    print(f"Saved: {report_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Logistic Regression on GalaxyMNIST.")
    parser.add_argument("--data-root", required=True, help="Dataset root (absolute path)")
    parser.add_argument("--output-dir", default=".", help="Folder for report and figure")
    args = parser.parse_args()
    results = run_logistic_regression(args.data_root, args.output_dir)
 
 
if __name__ == "__main__":
    main()