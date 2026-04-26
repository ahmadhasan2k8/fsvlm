## Summary

One-paragraph description of what this PR does and why.

## Type of change

- [ ] Bug fix
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] New dataset reader / benchmark contribution
- [ ] New baseline / comparison method
- [ ] Documentation only
- [ ] CI / tooling

## Testing

- [ ] `pytest tests/unit -q` passes locally
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy --ignore-missing-imports fsvlm/` passes (if applicable)
- [ ] Base-import speed still <2s (if relevant): `python -c "import time;t=time.time();import fsvlm;print(time.time()-t)"`
- [ ] New tests added for new behaviour

## Benchmark impact

If this changes training or evaluation code:

- [ ] I re-ran the affected sweep with ≥3 seeds
- [ ] Included the new result rows in `research/dataset_size_results.json`
- [ ] Numerical change is within float-precision tolerance (drift from prior seeds ≤ 0.005 AUROC)
- [ ] OR: explicitly flagged as a change that will produce different numbers; bumped the `recipe_version`

## Scope

Please confirm the change fits project scope (see `POSITIONING.md`):

- [ ] Does not add aerial/drone/wind/solar/utility-infrastructure datasets or examples
- [ ] Does not add medical-imaging datasets (regulated domain)
- [ ] Credits unsloth if the change touches training kernels

## Related issue

Closes #

## Notes for reviewers

Anything non-obvious reviewers should pay attention to? Any loose ends you're flagging intentionally?
