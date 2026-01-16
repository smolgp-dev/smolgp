
(install)=


# Installation Guide

As `smolgp` extends the functionality of `tinygp`, it utilizes a number of fundamental `tinygp` functions and classes and so requires `tinygp` as a dependency. 

:::{admonition} **Note: if not using uv** 
:class: tip
`smolgp` requires the latest version of the [tinygp GitHub repository](https://github.com/dfm/tinygp), rather than the version on PyPI (which is out of date on some aspects of the `quasisep` definitions). If you install `smolgp` with [uv](https://docs.astral.sh/uv/), this should be handled automatically. Otherwise, you will want to [install tinygp from source](https://tinygp.readthedocs.io/en/stable/install.html#from-source) in the environment which you wish to install `smolgp`.
:::

:::{admonition} **Installing on GPU** 
:class: tip
If you want to take advantage of the GPU-optimized parts of the code (e.g. the parallel solvers), you'll want to install the CUDA version of [`jax`](https://github.com/google/jax/#installation). You can do that with `uv` like so:
```bash
uv add smolgp[cuda]
```
or `uv add smolgp[cuda12]` or `uv add smolgp[cuda13]` for a specific version.
:::

## Using uv

The recommended way to install the most recent stable version of `smolgp` is to use [uv](https://docs.astral.sh/uv/):
```bash
uv add smolgp
```
To install in the current `uv` venv but not add it to `pyproject.toml`, you can always do
```bash
uv pip install smolgp
```

Of course, you can always just use [pip](https://pip.pypa.io):
```bash
python -m pip install smolgp
```

We do not recommend using `conda`.

## From source

Alternatively, you can get the source:

```bash
git clone https://github.com/smolgp-dev/smolgp.git
cd smolgp
uv pip install -e .
```
Or, to add to another `uv` enviornment:
```bash
uv add --editable /path/to/smolgp
```

## Tests

If you installed from source, you can run the unit tests. From the root of the
source directory, run:

```bash
uv sync --group test
uv run -m pytest 
```
Or in one line (can specify python version)
```bash
uv run --group test --python 3.13 -m pytest -n auto tests
```
To run a single test, e.g.
```bash
uv run pytest tests/test_kernels.py   
```
To run with coverage
```bash
uv run --group test --python 3.13 -m pytest --cov --cov-branch --cov-report=xml -n auto tests
```