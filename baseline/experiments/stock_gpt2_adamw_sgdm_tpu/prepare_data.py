"""Explicit preparation: download only the pinned original FineWeb GPT-2 shards."""

# Keep direct-script entry points inside this experiment's package namespace.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"
import argparse
from pathlib import Path

from .config import FULL_BUDGET_TOKENS
from .dataset import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--budget-tokens", type=int, default=FULL_BUDGET_TOKENS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.budget_tokens <= 0:
        parser.error("budget-tokens must be positive")
    result = prepare(args.data_dir, args.budget_tokens, download=not args.verify_only)
    print(f'Verified {len(result["files"])} pinned shards in {args.data_dir.resolve()}')


if __name__ == "__main__":
    main()
