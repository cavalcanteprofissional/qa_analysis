"""CLI entrypoint para a QA Pipeline."""
import argparse
from typing import List

from .pipeline_controller import PipelineController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA Pipeline")

    p.add_argument("--shards", type=str, nargs='+', default=["all"],
                   help="Shards to process: 'all', pattern 'shard_*.csv', or list")
    p.add_argument("--models", type=str, nargs='+', default=["all"],
                   help="Models to use: distilbert, roberta, bert, or all")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional)")

    return p.parse_args()


def main():
    args = parse_args()

    controller = PipelineController(
        shards=args.shards,
        models=args.models,
        batch_size=args.batch_size,
        workers=args.workers,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )

    controller.run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
