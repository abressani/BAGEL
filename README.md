BAGEL: Bayesian Analysis of Galactic Environments with machine Learning

# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
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
daneel -i <input_file> --transit [--output lightcurve.png]
daneel -i <input_file> --galaxy [--output galaxy.png]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-g, --galaxy`: Plot a GalaxyMNIST galaxy and its azimuthally averaged radial brightness profile
- `-o, --output`: Output path for the plot (default: `lc.png` for `--transit`, `galaxy.png` for `--galaxy`)
- `-d, --detect`: Reserved for the course detection exercise
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)

### Examples

```bash
# Run exoplanet detection
daneel -i examples/params.yaml --transit --output lightcurve.png

# Run atmospheric characterization
daneel -i examples/params.yaml --detect

# Run both detection and atmospheric analysis
daneel -i examples/params.yaml --atmosphere
```
## Galaxy morphology

```bash
daneel -i examples/params.yaml --galaxy --output galaxy.png
```

Plots one real GalaxyMNIST galaxy (currently train split, index 7) with its class
and azimuthally averaged radial brightness profile. The official GalaxyMNIST download
links no longer work, so the data are read from the Hugging Face mirror
`matthieulel/galaxy_mnist` and resized to 64x64. The first run downloads about 780 MB
into the Hugging Face cache and needs internet access.

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

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
