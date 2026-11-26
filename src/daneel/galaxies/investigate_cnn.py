"""Investigating what our CNN learned using Grad-CAM for the four-class 
GalaxyMNIST.

The class GalaxyExplainer imports from the GalaxyCNN classifier, so the
images are prepared and the network is rebuilt exactly as for the training.
The trained weights are loaded on the trained CNN outputs.

What it does:
  1. classify the official TEST set with the trained CNN
  2. write the lists of correctly classified and misclassified galaxies
  3. choose a few galaxies from each list with a rule given in the YAML
  4. draw the Grad-CAM map of each chosen galaxy

To run it (settings in the 'explain' section of the YAML file):
    python -m daneel.galaxies.investigate_cnn -i path_to_the_parameters.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from daneel.galaxies.ML_Classifier import GalaxyCNN
from daneel.galaxies.morphology import CLASS_NAMES

SELECTIONS = ("first", "random", "manual")

#choosing the galaxies
def choose_gal(indices, how, n, seed, manual=None):
    """Choose galaxies from a list of test-set indices with a rule chosen
     from the ones below:

    first  : the first n of the list (lowest indices)
    random : n at random, with a fixed seed (reproducible)
    manual : the indices given in the YAML (they must belong to the list)
    """    
    indices = np.asarray(indices)
    if how == "first":
        return indices[:n].tolist()
    if how == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(indices, size=min(n, len(indices)), replace=False).tolist())
    if how == "manual":
        manual = list(manual or [])
        outside = [i for i in manual if i not in set(indices.tolist())]
        if not manual or outside:
            raise ValueError(
                f"manual indices must be given and belong to the right list "
                f"(wrong ones: {outside})"
            )
        return manual
    raise ValueError(f"selection must be one of {SELECTIONS}")

#saving the gal image and its heatmap
def save_figure(samples, path):
    """Save the galaxies images on top row and their Grad-CAM map on bottom row.

    samples : list of dicts with image (64, 64, 3) in [0, 1], cam (64, 64) in
              [0, 1], text (title) and ok (True = correctly classified)
    """
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.8), squeeze=False)
    for k, s in enumerate(samples):
        colour = "tab:green" if s["ok"] else "tab:red"
        axes[0, k].imshow(s["image"])
        axes[0, k].set_title(s["text"], fontsize=8, color=colour)
        axes[1, k].imshow(s["image"])
        axes[1, k].imshow(s["cam"], cmap="magma", alpha=0.5, vmin=0, vmax=1)
        for row in (0, 1):
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
    axes[0, 0].set_ylabel("image")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

#to explain CNN--- maybe a class?

class GalaxyExplainer:
    """Grad-CAM using the exact results saved by GalaxyCNN."""

    def __init__(self, params):
        if params.get("results_file") is None:
            raise ValueError("'explain' section is missing: results_file")

        self.results_file = Path(params["results_file"]).expanduser()
        self.output_dir = Path(
            params.get("output_dir", self.results_file.parent)
        ).expanduser()

        self.selection = params.get("selection", "first")
        if self.selection not in SELECTIONS:
            raise ValueError(
                f"explain.selection must be one of {SELECTIONS}"
            )

        self.n_per_group = int(params.get("n_per_group", 2))
        self.selection_seed = int(params.get("selection_seed", 10))

        self.correct_indices = params.get("correct_indices")
        self.wrong_indices = params.get("wrong_indices")

        self.model = None
        self.channels = None
        self.class_names = None

    #load the saved cnn files
    def load_experiment(self):
        import torch

        if not self.results_file.is_file():
            raise FileNotFoundError(f"CNN experiment file not found: {self.results_file}")

        data = torch.load(self.results_file, map_location="cpu", weights_only=True)

        #recover exact cnn of input 
        config = data["config"]
        self.channels = tuple(data["channels"])
        self.class_names = tuple(data["class_names"])

        #rebuild the cnn using saved parameters
        cnn = GalaxyCNN({
            "data_root": ".",
            "channels": self.channels,
            "dropout": data["dropout"],
            "seed": config["seed"],
            "lr": config["lr"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "validation_fraction": config["validation_fraction"],
        })

        self.model = cnn.build_model()
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()

        #recover the original test-set data and predictions
        x = data["x_test"]
        y = data["y_test"].numpy()
        pred = data["y_pred"].numpy()
        probs = data["probabilities"].numpy()

        if not (len(x) == len(y) == len(pred) == len(probs)):
            raise ValueError("Inconsistent lengths in saved CNN experiment")

        return x, y, pred, probs

    #evaluating the Grad-CAM (heatmap)
    def grad_cam(self, image):
        """Return a (64, 64) Grad-CAM map in [0, 1] values."""

        import torch
        from torch.nn import functional

        k = 3 * len(self.channels) - 2
        maps = self.model[:k + 1](image[None])
        scores = self.model[k + 1:](maps)
        best = int(scores.argmax(dim=1))
        gradient = torch.autograd.grad(scores[0, best], maps)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * maps).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=image.shape[1:],
            mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        return cam / cam.max() if cam.max() > 0 else cam

    #save correct/wrong classified galaxies lists
    def save_lists(self, path, y, pred, probs, chosen_ok, chosen_bad):
        correct = np.flatnonzero(y == pred)
        wrong = np.flatnonzero(y != pred)

        lines = [
            "Grad-CAM of the CNN - Assignment 2, Task I",
            "=" * 60,
            f"Experiment file: {self.results_file.name}",
            f"Channels: {self.channels}",
            f"Official test set: {len(y)} galaxies; "
            f"correctly classified: {len(correct)}; "
            f"misclassified: {len(wrong)}; "
            f"accuracy: {len(correct) / len(y):.4f}",
            f"Selection: {self.selection}",
            f"Galaxies per group: {self.n_per_group}",
            f"Chosen correctly classified: {chosen_ok}",
            f"Chosen misclassified: {chosen_bad}",
            "",
        ]

        for title, group in (
            ("CORRECTLY CLASSIFIED", correct),
            ("MISCLASSIFIED", wrong),
        ):
            lines += [title, "test index, true class, predicted class, "
                "probability of predicted class"]

            lines += [
                f"{i}, {self.class_names[y[i]]}, "
                f"{self.class_names[pred[i]]}, "
                f"{probs[i, pred[i]]:.3f}"
                for i in group
            ]

            lines.append("")

        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    #running
    def run(self):
        x, y, pred, probs = self.load_experiment()
        correct = np.flatnonzero(y == pred)
        wrong = np.flatnonzero(y != pred)
        print(
            f"Test set: {len(correct)} correct, "
            f"{len(wrong)} misclassified "
            f"(accuracy {len(correct) / len(y):.4f})"
        )
        chosen_ok = choose_gal(correct, self.selection, self.n_per_group, self.selection_seed,
            self.correct_indices)
        chosen_bad = choose_gal(wrong, self.selection, self.n_per_group, self.selection_seed,
            self.wrong_indices)
        samples = []

        for ok, group in ((True, chosen_ok), (False, chosen_bad)):
            for i in group:
                samples.append({
                    "ok": ok,
                    "image": x[i].permute(1, 2, 0).numpy(),
                    "cam": self.grad_cam(x[i]),
                    "text": (
                        f"{'correct' if ok else 'wrong'} (test {i})\n"
                        f"true: {self.class_names[y[i]]}\n"
                        f"pred: {self.class_names[pred[i]]} "
                        f"({probs[i, pred[i]]:.2f})"
                    ),
                })

        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure = self.output_dir / "assignment2_taskI_explain.png"
        lists = self.output_dir / "cnn_explain_lists.txt"
        save_figure(samples, figure)
        self.save_lists(
            lists, y, pred, probs, chosen_ok, chosen_bad
        )
        print(f"Saved: {figure.resolve()}")
        print(f"Saved: {lists.resolve()}")

def main():
    from daneel.parameters import Parameters

    parser = argparse.ArgumentParser(description="Grad-CAM of the GalaxyMNIST CNN (Task I).")
    parser.add_argument("-i", "--input", dest="input_file", required=True,
                        help="Input parameter file (YAML) with an 'explain' section")
    args = parser.parse_args()
    section = Parameters(args.input_file).params.get("explain")
    if not isinstance(section, dict) or section.get("results_file") is None:
        parser.error("the input file must contain an 'explain' section "
        "with results_file"
        )
    GalaxyExplainer(section).run()

if __name__ == "__main__":
    main()