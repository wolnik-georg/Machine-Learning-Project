"""
Command-line tool for comparing ML experiments.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.experiment import (
    compare_experiments,
    find_best_experiment,
    generate_experiments_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Compare ML experiments.")
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run IDs to compare (e.g., --runs run_1 run_2)",
    )
    parser.add_argument(
        "--metric",
        default="test_accuracy",
        help="Metric to compare (default: test_accuracy)",
    )
    parser.add_argument(
        "--best",
        type=int,
        nargs="?",
        const=5,
        help="Show best N experiments (default: 5)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate summary of all experiments"
    )

    args = parser.parse_args()

    if args.summary:
        # Show summary of all experiments
        df = generate_experiments_summary()
        print("=== EXPERIMENTS SUMMARY === ")
        print(df.to_string(index=False))

    elif args.best:
        # Show best experiments
        df = find_best_experiment(metric=args.metric, top_k=args.best)
        print(f"=== BEST {args.best} EXPERIMENTS BY {args.metric} ===")
        print(df.to_string(index=False))

    elif args.runs:
        # Compare specified runs
        df = compare_experiments(run_ids=args.runs, metric=args.metric)
        print(f"=== COMPARING RUNS: {', '.join(args.runs)} BY {args.metric} === ")
        print(f"Metric: {args.metric}")
        print(df.to_string(index=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
