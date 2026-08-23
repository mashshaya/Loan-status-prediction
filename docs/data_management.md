# Data And Artifact Management

This repository keeps source code, tests, notebooks, and documentation in Git.
Large or potentially sensitive runtime artifacts are intentionally excluded:

- `Data/*.csv`
- `Data/*.zip`
- `models/*.joblib`
- `models/*.pkl`
- `reports/predictions.csv`

Recommended workflow:

1. Download or copy the raw dataset locally to `Data/loan_data.csv`.
2. Run training and reporting commands locally.
3. Commit code, tests, documentation, notebooks, and small synthetic examples.
4. Store large datasets and trained model binaries outside normal Git history.

If versioned datasets or model binaries become necessary later, use DVC or Git
LFS and document the storage location in this file.
