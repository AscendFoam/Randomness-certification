from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .mdi_single_device import run_route2, search_route2_high_entropy


def _clean_value(value: Any) -> Any:
    """Convert numpy scalars to JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: str | None, payload: Any) -> None:
    """Persist a JSON payload when requested."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_clean_value),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone runner for QRNG route 2.")
    parser.add_argument("--mode", choices=["baseline", "high-output-search"], default="baseline")
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-inputs", type=int, default=None)
    parser.add_argument("--num-outputs", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "baseline":
        result = run_route2(
            max_inputs_to_certify=args.max_inputs,
            preferred_solver=args.solver,
            verbose=args.verbose,
        )
    else:
        result = search_route2_high_entropy(
            num_outputs=args.num_outputs,
            num_trials=args.num_trials,
            preferred_solver=args.solver,
            seed=args.seed,
            verbose=args.verbose,
        )

    _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_clean_value))


if __name__ == "__main__":
    main()
