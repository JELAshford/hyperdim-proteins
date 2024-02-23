# Hyperdimensional Computation for Protein Family Classification

Use high-dimensional (~10_000) vector embeddings of amino-acids, combined using simple operators, to reliably classify protein amino-acid subsequences into functionally/evolutionarily related families.

## Setup

### Python Environment

Create a `Python` environment with the flavour of your choosing *(`venv`, `conda`, etc.)*, activate it, and install the module in this local repo with:

```bash
maturin develop --release
python -m pip install -e .
```

### Data Source

The data from this project can be acquire from [this Kaggle repo](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split), and should be first extracted and included in a folder called `/data` in this repository, such that `tree .` includes:

```bash
.
├── README.md
├── data
│   ├── dev
│   ├── test
│   └── train
├── docs
...
```

then run `python process_data.py` to convert each of those folders in to a single `.feather` file which will then be used in subsequent analysis.
