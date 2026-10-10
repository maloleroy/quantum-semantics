"""Export measured fit costs and four-slot projections for another same-arm batch."""

import argparse
from pathlib import Path

from qcse.benchmark_compute_report import timing_report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    cli.add_argument("--fits-per-arm", type=int)
    cli.add_argument("--slurm-accounting", type=Path)
    args = cli.parse_args()
    timing_report(args.study.resolve(), args.concurrency, args.fits_per_arm, args.slurm_accounting)
    print(args.study / "timing-report" / "report.md")


if __name__ == "__main__":
    main()
