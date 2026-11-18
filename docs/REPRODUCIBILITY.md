# Reproducing the audit

## Recorded evidence versus a new run

The recorded run key is `653b00226ec3d41dec2775b67bd2fd9dd9839f1ccd2040782f3c9375c8c62437`. Its original configuration, package versions, source hashes, and phase status are preserved under `results/recorded/manifests/` and `configs/recorded-paper.json`.

`configs/paper.json` retains the scientific settings but resets the checkpoint path to `null`, the working directory to `runs`, and the requested phase range to 1–10. File paths and execution scope are runtime settings. An environment change can produce a different run identity and invalidate cached tasks; matching versions does not guarantee bitwise equality on different hardware.

## Kaggle

1. Import `notebooks/environmental_forecasting.ipynb` or open the original Kaggle notebook.
2. Enable Internet and select CPU execution. The documented models do not require a GPU.
3. For a fresh run, set `RESUME_FROM=None`, `START_PHASE=1`, and `END_PHASE=2`.
4. Run all cells. Download the partial ZIP generated at the end of the requested phase range.
5. For later sessions, attach that partial archive and set `RESUME_FROM` to its exact path. Set `END_PHASE=5`, then 8, then 9, and finally 10. Run from the top each time; compatible prerequisite tasks load from cache.

The supplied notebook used `AUTO_KEEP`, a project-specific convention for finding a renamed checkpoint ZIP. A new user should use `None` for a fresh run or supply an explicit valid checkpoint path. The repository notebook changes only that default, clears saved outputs, and removes saved execution/attachment metadata. Embedded scientific modules remain byte-identical to the archived source.

The final output ZIP is a dissemination package, not a resumable checkpoint archive. Keep partial archives if continuation is needed. The documented staged schedule reflects the source notebook; runtime depends on hardware and the amount of reusable computation.

## Local execution

Create and activate a Python environment, then install:

```bash
python -m pip install -r requirements-publication.txt
python scripts/run_audit.py --end-phase 2
python scripts/run_audit.py --end-phase 5
python scripts/run_audit.py --end-phase 8
python scripts/run_audit.py --end-phase 9
python scripts/run_audit.py --end-phase 10
```

Keep `runs/` between invocations so compatible checkpoints remain available. The runner invokes every prerequisite phase in order and leaves cache reuse to the original driver. To resume from a different extracted run directory, pass `--resume /absolute/path/to/run`; ZIP extraction is handled by the notebook, not the local runner.

Use `--config path/to/config.json` for explicit offline input paths, or `--work-root path/to/runs` to change the working directory. Relative paths are resolved from the current shell directory; run the examples from the repository root. Thread limits are set before scientific modules are imported.

The original notebook also provides a `pilot` profile. Pilot results are engineering checks and must not be presented as the paper's results. The supplied local configuration is the paper profile.

## Phase map

| Phase | Work |
| --- | --- |
| 1 | Input loading, historical splits, causal features, and checks |
| 2 | Development-only temporal cross-validation and clean models |
| 3 | RQ1: missingness |
| 4 | RQ2: sensor faults |
| 5 | RQ3: explanation stability |
| 6 | RQ4: drift detection and adaptation |
| 7 | RQ5: intervals and decisions |
| 8 | RQ6: computation and measurement provenance |
| 9 | External replication and all-station Beijing evaluation |
| 10 | Artifact regeneration, recorded checks, evidence gates, and final ZIP |

## Verification performed for this repository

- All 2,099 original checksum-manifest entries matched the uploaded archive.
- All twelve Python source modules matched the notebook's embedded module strings.
- Scientific modules and repository scripts were syntax-checked.
- Repository notebook structure and local runner argument handling were checked.

No full scientific rerun or new raw-data download was performed. The archive reports 13 passed regression checks; this report is preserved as recorded evidence. Recorded requirements are not a guarantee that every pinned package has a wheel for every operating system or Python version. PyMuPDF was used by the original PDF check but omitted from its package-version record, so its version remains unpinned.
