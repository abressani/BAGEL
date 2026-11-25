BAGEL: Bayesian Analysis of Galactic Environments with machine Learning

# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling galaxy morphology classification (GalaxyMNIST) and Baryonic Tully–Fisher Relation (BTFR) analysis (SPARC/BayesLineFit and dynesty). The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch (`torch`) for the CNN classifier (`-d cnn`) and the GAN (`--dream`)

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_26.git
cd comp_astro_26
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --galaxy [--index N] [--output galaxy.png]
daneel -i <input_file> -d rf
daneel -i <input_file> -d cnn
daneel -i <input_file> --dream
```

All the paths are read from the input file, so the command can be run from any
directory, including one outside the repository.

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-g, --galaxy`: Plot a GalaxyMNIST galaxy and its azimuthally averaged radial brightness profile
- `-d, --detection {rf,cnn}`: Train and test a classifier on GalaxyMNIST: `rf` = Random Forest, `cnn` = convolutional neural network (PyTorch)
- `--dream`: Train a generative adversarial network (GAN) on GalaxyMNIST and generate synthetic galaxies
- `-o, --output`: Output path for the `--galaxy` plot (default: `galaxy.png`)
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `--index`: Zero-based index of the galaxy to plot with `--galaxy`. It is only used when the `galaxy` section of the input file has no `index`. Valid range: 0–7999 for the train split

### Examples

```bash
# Galaxy morphology
daneel -i examples/params.yaml --galaxy --output galaxy.png

# Run the Random Forest classifier
daneel -i examples/params.yaml -d rf

# Run the convolutional neural network
daneel -i examples/params.yaml -d cnn

# Generate synthetic galaxies with a GAN
daneel -i examples/params.yaml --dream
```

## Galaxy morphology

`--galaxy` plots one real GalaxyMNIST galaxy with its class and its azimuthally
averaged radial brightness profile. The dataset folder, the split and the galaxy index
are read from the `galaxy` section of the YAML file (see below). The images are
read from `<data_root>/GalaxyMNIST/raw/<split>_dataset.hdf5`.

If that file is not found, the data are read from the Hugging Face mirror
`matthieulel/galaxy_mnist` instead (the official GalaxyMNIST download links no longer
work) and resized to 64x64. This ignores `data_root`, needs internet access, and the
first run downloads about 780 MB into the Hugging Face cache.

The output image is saved as `output_dir/output` if the YAML file gives them,
otherwise as the `-o/--output` path (default `galaxy.png`).

### Random Forest classifier

```bash
daneel -i path_to_the_parameters.yaml -d rf
```

This classifies the galaxies into the four morphology classes with a Random Forest.
For each galaxy, the features are the radial brightness profile and the mean R, G, B colour.
Several Random Forest configurations are trained and compared on a validation set
(a stratified part of the train set), and the best one is then evaluated once on the test set.

All the settings are read from the `random_forest` section of the YAML file.

### Convolutional neural network

```bash
daneel -i path_to_the_parameters.yaml -d cnn
```

This classifies the galaxies into the same four morphology classes with a PyTorch
CNN trained directly on the 64x64 RGB images. The network has one block (3x3
convolution, ReLU, 2x2 max pooling) per entry of `channels`, followed by global
average pooling, dropout and a linear layer with four outputs. It is trained with
Adam and cross-entropy loss on a stratified part of the official train set; the
rest is the validation set. The reported metrics (accuracy, macro-averaged
precision and recall, and the 4x4 confusion matrix) are those of the final epoch.
Every run trains from scratch with the given seed, so the results are reproducible.

If `evaluate_test` is true, the official test set is evaluated as well. The test
set must not be used to choose a model, so set it to true only for the final,
fixed settings.

All the settings are read from the `cnn` section of the YAML file. The output, in
`output_dir`, is:

- `cnn_report.txt`: settings, metrics and confusion matrices
- `cnn_confusion_matrix_validation.png`
- `cnn_confusion_matrix_test.png` (if `evaluate_test` is true)
- `cnn_weights.pt`: the trained weights

daneel sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing PyTorch, which avoids an
`OMP: Error #15` crash that torch can cause on some macOS conda setups.

### Dreaming new galaxies (GAN)

```bash
daneel -i path_to_the_parameters.yaml --dream
```

This trains a generative adversarial network (a DCGAN) on the real GalaxyMNIST images
and uses it to create new, synthetic galaxy-like RGB images. The generator turns a
random noise vector into a 64x64 RGB image through four transposed-convolution
blocks. The discriminator, a stack of strided convolutions, learns to tell real
images from generated ones, and the two networks are trained against each other
(binary cross-entropy loss, Adam optimiser, random flips of the real images). The
model is unconditional: it does not use the galaxy classes.

All the settings are read from the `dream` section of the YAML file. The output, in
`output_dir`, is:

- `assignment2_taskI_gan.png`: grid of generated galaxies
- `assignment2_taskI_real.png`: grid of real galaxies, for comparison
- `report_gan_assignment2_taskI.txt`: training setup, a crude real-vs-generated
  comparison and the limitations of the synthetic images

Training takes a while on a CPU. For a quick test, use a few `epochs` and
`base_channels: 32`. The model lives in the reusable module `daneel.dream`:

```python
from daneel.dream import GalaxyGAN

gan = GalaxyGAN({"data_root": "/absolute/path/to/data", "epochs": 10})
gan.train(gan.load_images())
galaxies = gan.generate(16)      # array (16, 64, 64, 3) with values in [0, 1]
```

## Input File Format

The input file is a YAML file with one section for each analysis (`galaxy`, `random_forest`, `cnn`, `dream`).
Each command reads only its own section, so one file can hold several of them.

For the galaxy plot:

```yaml
galaxy:
  data_root: /path/to/data/          # folder containing GalaxyMNIST/raw/ (absolute path)
  split: train                       # train or test (default: train)
  index: 42                          # galaxy index; if missing, --index is used
  output: galaxy.png                 # optional, output file name
  output_dir: /path/to/output        # optional, folder of the output file
```

For the Random Forest:

```yaml
random_forest:
  data_root: /path/to/data/          # folder containing GalaxyMNIST/raw/ (absolute path)
  output_dir: /path/to/output        # where report and figure are saved
  seed: 10                           # random seed
  validation_size: 0.2               # fraction of the train set used for validation
  features:
    n_profile_bins: 32               # number of radial profile bins used as features
    use_colour: true                 # add the mean R, G, B colour to the features
  hyperparameters:                   # configurations to compare
    - {n_estimators: 50,  max_depth: 5}
    - {n_estimators: 100, max_depth: 10}
    - {n_estimators: 200, max_depth: 10}
    - {n_estimators: 200, max_depth: null}   # null = no depth limit
```

For the Convolutional Neural Network:

```yaml
cnn:
  data_root: /path/to/data/          # folder containing GalaxyMNIST/raw/ (absolute path)
  output_dir: /path/to/output        # report, figures and weights; created if missing
  channels: [16, 32, 64]             # filters of each convolutional block
  dropout: 0.0                       # dropout before the output layer
  lr: 0.001                          # Adam learning rate
  epochs: 60
  batch_size: 32
  validation_fraction: 0.25          # fraction of the train set used for validation
  seed: 84631                        # random seed (split and weights)
  evaluate_test: true                # official test set; only for the final, fixed settings
```

Only `data_root` is required. The other keys default to the values above, except
`output_dir` (current directory) and `evaluate_test` (false).

For the GAN:

```yaml
dream:
  data_root: /path/to/data/          # folder containing GalaxyMNIST/raw/ (absolute path)
  output_dir: /path/to/output        # figures and report; created if missing
  split: train                       # real images used for training: train or test
  epochs: 50
  batch_size: 64
  latent_dim: 100                    # length of the noise vector
  base_channels: 64                  # width of both networks (32 = smaller and faster)
  lr: 0.0002                         # Adam learning rate
  beta1: 0.5                         # Adam first momentum
  seed: 84631                        # random seed (weights, batches, noise of the grid)
  grid_size: 8                       # the figure shows 8 x 8 = 64 galaxies
  augment: true                      # random flips of the real images
  device: cpu                        # cpu, cuda or mps
  # num_threads: 1                 # CPU threads; by default torch's setting is left unchanged
```

Only `data_root` is required. The output file names can be changed with the optional
keys `figure`, `real_figure` and `report`.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
