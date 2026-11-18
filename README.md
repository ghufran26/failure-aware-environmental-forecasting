# Failure-Aware Environmental Forecasting

Research code and recorded outputs accompanying **Failure-Aware Environmental Forecasting Under Missing, Faulty, and Shifted Sensor Data**.

[Kaggle notebook](https://www.kaggle.com/code/ghufran26/fork-of-primna-tom-paper-v3) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Datasets](data/README.md) · [Results](results/README.md)

## Overview

This study evaluates environmental forecasting when sensor measurements are missing, faulty, or shifted. A shared temporal design separates development, calibration, policy selection, and testing. The audit examines forecast error, explanation stability, drift alarms, adaptive updating, uncertainty, decision costs, and computational burden.

The core applications are hourly carbon-monoxide forecasting and ten-minute appliance-energy forecasting. External retrospective replication uses Beijing monitoring stations and household electricity measurements, including leave-one-station-out evaluation across all twelve Beijing stations.

| Question | Evaluation |
| --- | --- |
| RQ1 | Five missingness mechanisms and mechanism-specific degradation |
| RQ2 | Six sensor faults and average versus upper-tail forecast error |
| RQ3 | Grouped permutation explanations against paired clean-repeat controls |
| RQ4 | False-alarm feasibility and alarm, scheduled, guarded, and budget-matched updating |
| RQ5 | Empirical interval coverage, selective prediction, and cost/capacity sensitivity |
| RQ6 | Training time, inference time, memory, and energy-measurement provenance |

## Start here

- **Inspect the evidence:** browse [publication tables](results/recorded/tables), [vector figures](results/recorded/figures_pdf), and [numeric plot data](results/recorded/plot_data).
- **Run on Kaggle:** open the linked notebook or import [the repository notebook](notebooks/environmental_forecasting.ipynb). Enable Internet, select CPU, and use `RESUME_FROM=None` for a fresh run.
- **Run locally:** install dependencies and use the command below. Start with phases 1–2 before attempting the full audit.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements-publication.txt
python scripts/run_audit.py --end-phase 2
```

The recorded run used Python 3.13.15, four CPU threads, and master seed `20260911`. Exact recorded dependency versions are in [requirements.txt](requirements.txt). PyMuPDF is additionally required for final PDF checks; its original version was not recorded. See the reproduction guide before continuing through phase 10. A full scientific rerun has not been performed during repository preparation.

## Repository layout

```text
notebooks/        Kaggle-compatible notebook, cleared of saved outputs
src/              Twelve unchanged modules from the completed output archive
configs/          Recorded configuration and a fresh-run configuration
scripts/          Local runner and archive integrity verifier
data/             Official sources, expected files, and acquisition notes
results/recorded/ Publication figures, tables, plot data, and original manifests
docs/             Reproduction and GitHub upload instructions
release/          Full-output archive description and original ZIP checksum
```

## Complete experimental outputs

The original `primna-tom-res.zip` is a separate release asset. It contains the detailed numeric results omitted from this lightweight repository. See [release/README.md](release/README.md) for verification and upload instructions. The release asset has **not yet been published**.

The supplied run records phases 1–10 as complete and reports 13 regression checks passed. Repository preparation independently verified all 2,099 entries in the original output-checksum manifest and confirmed that all twelve source modules match the supplied notebook. These are integrity checks, not a new execution of the experiments.

## Interpretation and scope

Both core evaluations and the external replication are retrospective. Corruption replicates are repeated interventions on fixed histories, not independent sites. The archived evidence does not support the target rare false-alarm rate; prediction-interval coverage is empirical under temporal dependence and shift. Energy telemetry was unavailable, so timing and memory should not be interpreted as measured energy or carbon results.

## Citation and licensing

See [CITATION.md](CITATION.md). Publication identifiers, the complete author list, and a reuse license are pending author confirmation. Dataset terms remain those of their respective providers. No paper acceptance or publication status is asserted.
