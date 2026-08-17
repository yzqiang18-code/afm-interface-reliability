# Tests

Run all repository tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The suite covers cohort preparation, metric helpers, chain-exchange sentinel
handling, consistency and physics metrics, leakage-audit helpers, full-precision
AF-M ranking extraction, and the Candidate Ridge train/predict/evaluate path.
`scripts/validate_repository.py` adds artifact, hash, schema, link, and
scientific release-gate checks.
