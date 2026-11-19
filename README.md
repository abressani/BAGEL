BAGEL: Bayesian Analysis of Galactic Environments with machine Learning

# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling galaxy morphology classification (GalaxyMNIST) and Baryonic Tully–Fisher Relation (BTFR) analysis (SPARC/BayesLineFit and dynesty). The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10

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
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-g, --galaxy`: Plot a GalaxyMNIST galaxy and its azimuthally averaged radial brightness profile
- `-d, --detection rf`: Train and test a Random Forest classifier on GalaxyMNIST
- `-o, --output`: Output path for the plot (default: `lc.png` for `--transit`, `galaxy.png` for `--galaxy`)
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)
- `--index`: Zero-based index of the galaxy to plot with `--galaxy` (default: 7). Valid range: 0–7999 for the train split

### Examples

```bash
## Galaxy morphology
daneel -i examples/params.yaml --galaxy --output galaxy.png

# Run the Random Forest classifier
daneel -i examples/params.yaml -d rf
```

## Galaxy morphology

`--galaxy` plots one real GalaxyMNIST galaxy (currently train split, index 7) with its class
and azimuthally averaged radial brightness profile. The official GalaxyMNIST download
links no longer work, so the data are read from the Hugging Face mirror
`matthieulel/galaxy_mnist` and resized to 64x64. The first run downloads about 780 MB
into the Hugging Face cache and needs internet access.

### Random Forest classifier

```bash
daneel -i path_to_the_parameters.yaml -d rf
```

This classifies the galaxies into the four morphology classes with a Random Forest.
For each galaxy, the features are the radial brightness profile and the mean R, G, B colour.
Several Random Forest configurations are trained and compared on a validation set
(a stratified part of the train set), and the best one is then evaluated once on the test set.

All the settings are read from the `random_forest` section of the YAML file.


## Input File Format

The input file is a YAML file with one section for each analysis (`transit`, `galaxy`, `random_forest`).
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
