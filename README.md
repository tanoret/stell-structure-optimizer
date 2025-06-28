# Stell Structure Optimizer

Analyzer and Optimizer for steel frame structures built using standard commercial profiles.

## Features
- Finite element frame analysis
- AISC 360 LRFD verification
- Streamlit based web application for interactive modelling
- Example notebooks demonstrating optimization workflows

## Installation
Install the package from the repository root:

```bash
pip install .
```

For development work you may prefer an editable install:

```bash
pip install -e .
```

## Running the tests

```bash
pytest -q
```

## Launching the web application

Use Streamlit to start the app:

```bash
streamlit run src/stell_structure_optimizer/app/app.py
```

The application loads profile data from the `database/` directory and lets you build or import structural models for analysis.

## Additional resources

Example notebooks are located in `examples/` and demonstrate common analysis and optimisation tasks.
