#!/usr/bin/env python3
"""Generate a synthetic dataset in the expected schema (default: data/)."""
import argparse

from ipo_model.data.synthetic import generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data")
    p.add_argument("--n-ipos", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    frames = generate(args.out, n_ipos=args.n_ipos, seed=args.seed)
    print(f"Wrote {len(frames['ipos'])} IPOs, {len(frames['prices'])} price rows, "
          f"{len(frames['gpr'])} GPR days to {args.out}/")


if __name__ == "__main__":
    main()
