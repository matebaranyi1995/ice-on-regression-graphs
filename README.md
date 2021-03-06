# ICE on regression graphs (and DAGs)

Python implementation of the ICE algorithm on regression graphs based on the paper 

[M. Baranyi, M. Bolla, *Iterated Conditional Expectation algorithm on DAGs and regression graphs*, Econometrics and Statistics, 2020](https://doi.org/10.1016/j.ecosta.2020.05.003 )

In its current state, the code belongs to the initial state of the project but fixes were applied to work with Python 3.8 inside the Conda environment built by the provided `.yml` file. You can create a conda environment by

`conda env create -f ICE_env.yml`

Then activate the environment:

`conda activate ICE_test`

A little framework in DASH is provided to test the algorithm with a user-provided tabular data file (preferable in `.xlsx` format). You can start the dashboard by

`python app.py`

The dashboard is not compatible with every attributes of the model.

A jupyter notebook is also available to test a code: `ICE_example.ipynb`

Building the regression graph (structure learning) with the provided R script has two R package dependencies:
- `gRchain`: available on [R-Forge](https://r-forge.r-project.org/R/?group_id=2099)
- `jsonlite`: available on CRAN
The R script and the wrapper function worked for me under R 4.0.2.

More detailed description of the codes may be available later on. 
