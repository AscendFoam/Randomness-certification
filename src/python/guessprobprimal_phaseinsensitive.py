"""Compatibility wrapper for the repaired route-4 phase-insensitive runner."""

import sys

from qrng_routes.route4.main import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "primal-single"])
    main()
