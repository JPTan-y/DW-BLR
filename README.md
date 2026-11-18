# DW-BLR

Code and data for **DW-BLR**, a decorrelation-weighted Bayesian logistic regression with spike-and-slab variable selection, and for its comparison with Lasso, Elastic Net, SCAD, SRDO, SVI, DWR and VBLR in a simulation study and a multi-cohort breast cancer pCR application.

## Repository layout

```
DWBLR/
├── experiment.py              # Fit / evaluate all methods with a common interface
├── function/                  # Public wrappers (Lasso, ElasticNet, SCAD, SRDO, SVI, DWR, VBLR, DW_BLR)
│   ├── real_data.py           # Real-data helpers 
│   └── real_data_penalties.R  # Lasso / Elastic Net / SCAD fits with glmnet and ncvreg
├── _backends/                 # Method implementations
├── simulation/
│   ├── generate_simulation_data.py
│   ├── parameters.py          # Simulation hyperparameters
│   ├── seed/seed_table.csv    # seeds 
│   ├── simulation_comparison.ipynb
│   └── test_simulation.py
├── example/
│   ├── parameters.py          # Real-data hyperparameters
│   ├── real_data.ipynb
│   ├── install_R_packages.R
│   └── data/                  # Preprocessed expression data and clinical subgroup labels
├── requirements.txt
└── original_runtime.json      
```

## Requirements

### Python

**Python 3.10** is required (the pinned `torch==1.12.1` has no wheels for Python 3.11 or later).

```bash
conda create -n dwblr python=3.10
conda activate dwblr
pip install -r requirements.txt
```


### R (real-data analysis only)

The Lasso, Elastic Net and SCAD baselines in the real-data analysis are fitted in R with exact package versions:

| Package  | Version |
|----------|---------|
| glmnet   | 4.1-10  |
| ncvreg   | 3.16.0  |
| jsonlite | any     |

Install them into a user library:

```bash
export R_LIBS_USER=/path/to/your/R/library      # Windows: set R_LIBS_USER=C:\path\to\R\library
Rscript example/install_R_packages.R
```

The notebook finds `Rscript` through the `RSCRIPT` environment variable, then `PATH`, then (on Windows) the registry. Set `RSCRIPT` if R is installed elsewhere. The simulation study does not need R.

## Running

### Simulation study

```bash
python simulation/test_simulation.py            # sanity checks for the data generator
jupyter lab simulation/simulation_comparison.ipynb
```

Data are generated in memory from `simulation/seed/seed_table.csv` (training at r = 2; test environments r ∈ {±1.5, ±2, ±2.5, ±3}; n_train ∈ {1500, 3000, 5000}; 100 replications). Results, per-replication checkpoints and figures are written to `simulation/simulation_results/`. Set `N_JOBS` in the first cell to run replications in parallel. The notebook records a fingerprint of the code and settings and refuses to mix results from different configurations in one results directory.

### Real-data analysis

```bash
jupyter lab example/real_data.ipynb
```

Run it from the repository root or from `example/`. Outputs (models, tables, figures and a `manifest.json`) are written to `example/results/real_data_<fingerprint>/`.

## Data

`example/data/data.xlsx` contains the preprocessed expression values of 100 genes and binary pCR outcomes (pCR = 1, residual disease = 0) for one training cohort and six test cohorts, all from the Gene Expression Omnibus:

| Cohort | Role | n | pCR |
|--------|------|---|-----|
| GSE25066 (GSE25055 + GSE25065) | Training | 466 | 93 |
| GSE163882 | Test | 147 | 48 |
| GSE41998  | Test | 108 | 30 |
| GSE230881 | Test | 171 | 22 |
| GSE140494 | Test | 69  | 16 |
| GSE194040 | Test | 175 | 27 |
| GSE231629 | Test | 103 | 10 |

The workbook holds the processed data used by the models, not the raw GEO series. `clinical_subgroups.csv` holds the HR/HER2 subgroup labels used for the subgroup ROC analysis; the rules are documented in `clinical_subgroups_provenance.json`. Please cite the original GEO studies when reusing these data.


