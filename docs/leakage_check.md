# Leakage Check

The strongest risk found in the current dataset is
`previous_loan_defaults_on_file`.

Observed target rates:

| previous_loan_defaults_on_file | Rows | Target rate |
| --- | ---: | ---: |
| No | 22,142 | 0.4516 |
| Yes | 22,858 | 0.0000 |

This is suspicious because one category perfectly maps to the negative class in
the current data. The feature may still be valid if it is genuinely known before
the loan decision, but it needs explicit business confirmation.

Until confirmed, report model quality both with and without this feature. The
project includes a reproducible command:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.leakage
```

The command writes `reports/leakage_check.json` with target-rate diagnostics and
a baseline comparison.
