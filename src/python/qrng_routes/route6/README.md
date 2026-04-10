# Route 6

Route 6 is a "new route3"-style prototype with four changes:

1. The trusted coherent alphabet allows multiple radii.
2. The output discretization uses route5-style axis-aligned IQ rectangles.
3. Output probabilities are computed from analytic Gaussian rectangle integrals.
4. Trusted input states are represented in the exact coherent-alphabet support via the Gram matrix.

Main entrypoints:

- `python -m qrng_routes.route6 --mode single`
- `python -m qrng_routes.route6 --mode partition-search`
- `python -m qrng_routes.route6 --mode alphabet-search`
- `python -m qrng_routes.route6 --mode fixed-partition-alphabet-search`

Example:

```bash
PYTHONPATH=src/python python -m qrng_routes.route6 \
  --mode single \
  --radius-values 0 0.6 1.2 \
  --phase-values 0 1.5707963267948966 3.141592653589793 4.71238898038469 \
  --num-x-bins 2 \
  --num-p-bins 2 \
  --quadrature-range 3.0
```

Second-round style fixed-partition scan:

```bash
PYTHONPATH=src/python python -m qrng_routes.route6 \
  --mode fixed-partition-alphabet-search \
  --radius-values 0.4 0.8 1.2 1.6 \
  --phase-values 0 1.5707963267948966 3.141592653589793 4.71238898038469 \
  --num-radii-values 2 3 4 \
  --num-phase-values 2 4 \
  --num-x-bins 6 \
  --num-p-bins 2 \
  --quadrature-range 1.5 \
  --boundary-gamma 1.5 \
  --min-local-states 6 \
  --max-local-states 8 \
  --max-inputs-to-certify 0
```

Here `--max-inputs-to-certify 0` means: certify all input pairs for each alphabet candidate instead of only the raw-best target.
