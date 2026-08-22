# Validation record

Validated on 2026-08-21 with Python 3 and the declared NumPy/SciPy runtime dependencies.

## Automated tests

Command:

```bash
python tests/run_tests.py
```

Result: **10 tests passed**.

Coverage includes:

- analytic objective and nonlinear-cost gradients versus centered finite differences;
- live progress-callback initial/final state and final-result consistency;
- PGD holdings and objective versus the exact quadratic KKT solution;
- exact KKT versus SciPy SLSQP;
- nonlinear power-law PGD versus SciPy SLSQP;
- Dykstra projection feasibility and idempotence;
- long-only, caps, affine equalities, sector inequalities, hard turnover;
- long-short dollar/beta neutrality, box bounds, and gross exposure.

## Notebook execution

Command:

```bash
python scripts/execute_notebooks.py
```

Result: every code cell and every embedded numerical assertion passed in all three notebooks.

## Packaging smoke test

The project was built as a PEP 517 wheel with local build dependencies, installed into a clean
temporary target, imported from that installed target, and used to run `examples/quick_start.py`.
The example converged with constraint violation below `1e-9`.
