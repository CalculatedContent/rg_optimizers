from __future__ import annotations

import argparse
from pathlib import Path

from rg_nanogpt_one_head.data import (
    TOKEN_DTYPE,
    load_memmaps,
    prepare_fineweb_edu,
    validate_prepared_data,
    write_token_splits,
)

from .config import load_config, roots

__all__ = [
    "TOKEN_DTYPE",
    "load_memmaps",
    "prepare_fineweb_edu",
    "validate_prepared_data",
    "write_token_splits",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned FineWeb-Edu corpus for NGB v4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output_dir) if args.output_dir else roots(cfg)["data"]
    prepare_fineweb_edu(cfg, output, force=args.force)


if __name__ == "__main__":
    main()
