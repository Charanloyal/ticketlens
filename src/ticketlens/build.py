"""Production build for the serving model (used by Render, Docker and CI).

Runs each step in its own Python process so that only one step's memory is resident at a time:
  1. download the Hugging Face dataset (fails instead of falling back to the tiny fixture)
  2. clean, split and augment it into a compact text/label corpus
  3. train the TF-IDF + Logistic Regression model from that corpus

Usage: python -m ticketlens.build [--mode all] [--max-peak-mb 512]
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

try:  # resource is POSIX-only (Render, Docker and CI run Linux)
    import resource
except ImportError:  # pragma: no cover - Windows development machines
    resource = None


def _steps(mode: str) -> List[Tuple[str, List[str]]]:
    return [
        ("download dataset", ["-m", "ticketlens.data", "--download-only", "--require-dataset"]),
        ("prepare training corpus", ["-m", "ticketlens.data", "--prepare-corpus", "--mode", mode, "--require-dataset"]),
        ("train baseline model", ["-m", "ticketlens.train_baseline", "--from-corpus", "--mode", mode]),
    ]


def _children_peak_mb() -> Optional[float]:
    if resource is None:
        return None
    # ru_maxrss of RUSAGE_CHILDREN is the peak RSS of the largest finished child, in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024


def _annotate(level: str, message: str) -> None:
    prefix = f"::{level}::" if os.environ.get("GITHUB_ACTIONS") == "true" else f"[build] {level.upper()}: "
    print(prefix + message, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="all")
    parser.add_argument(
        "--max-peak-mb",
        type=float,
        default=None,
        help="Fail if any step's peak memory exceeds this many MB (Linux only)",
    )
    args = parser.parse_args(argv)

    previous_peak = 0.0
    for name, step_args in _steps(args.mode):
        print(f"[build] >>> {name}", flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, *step_args], check=False)
        if result.returncode != 0:
            _annotate("error", f"Build step '{name}' failed with exit code {result.returncode}")
            return result.returncode

        duration = time.time() - started
        peak = _children_peak_mb()
        if peak is None:
            print(f"[build] <<< {name} done in {duration:.1f}s", flush=True)
            continue
        # The children peak is a running maximum, so it only identifies a step when it grows.
        step_peak = f"{peak:.0f} MB" if peak > previous_peak else f"<= {previous_peak:.0f} MB"
        previous_peak = max(previous_peak, peak)
        _annotate("notice", f"Build step '{name}' done in {duration:.1f}s, peak memory {step_peak}")
        if args.max_peak_mb is not None and peak > args.max_peak_mb:
            _annotate("error", f"Build step '{name}' peaked at {peak:.0f} MB (> {args.max_peak_mb:.0f} MB)")
            return 1

    print("[build] serving model ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
