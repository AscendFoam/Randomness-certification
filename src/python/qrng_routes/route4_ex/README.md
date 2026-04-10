# Route4-ex

`route4-ex` is a minimal prototype for the "route4 with non-diagonal trusted inputs" direction.

Current scope:

1. Trusted inputs are exact truncated coherent states `|alpha><alpha|` in the Fock basis.
2. It currently supports two built-in probability backends:
   - a toy binary coherent-projector POVM;
   - a more APD-like displaced-count histogram model with coarse-grained outputs, including detection efficiency and dark counts.
3. It also supports injecting an external probability table from `.mat`, `.npy`, `.npz`, or `.json`.
4. The code compares:
   - a `diagonal primal` baseline that only allows Fock-diagonal POVM elements;
   - a `full primal` model that allows general Hermitian PSD POVM elements.

This is intentionally a structural prototype, not yet an experimental-data pipeline.

APD-like example:

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex \
  --mode apd-compare \
  --alpha-values 0.6+0j 0+0.6j 0.4+0.4j \
  --displacement-alpha 0.35+0.35j \
  --num-outputs 4 \
  --raw-num-bins 16 \
  --detection-efficiency 0.6 \
  --dark-count-mean 0.02 \
  --cutoff 12 \
  --solver SCS
```

External table example:

```bash
PYTHONPATH=src/python python -m qrng_routes.route4_ex \
  --mode external-compare \
  --alpha-values 0.6+0j 0+0.6j -0.6+0j \
  --external-probability-path src/matlab/Probability.mat \
  --external-variable-name Probability \
  --external-row-indices 0 1 2 \
  --num-outputs 4 \
  --cutoff 12 \
  --solver SCS
```

The main diagnostic is whether `full primal` and `diagonal primal` begin to differ once the trusted inputs are non-diagonal.

If an amplitude starts with `-`, quote it or pass it from Python directly, otherwise the shell may parse it as a new option.
