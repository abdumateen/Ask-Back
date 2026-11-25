"""§14.2 `run`: execute A1's cell grid from a §14.1 config.

Idempotent and resumable: episodes already present under the same config hash
are skipped, so re-running after an interruption costs only the missing ones.
"""

from __future__ import annotations

import argparse
import sys

from askback.runner.cli import effective_config, load_config, run_study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m askback.run")
    ap.add_argument("--config", required=True, help="§14.1 YAML config")
    ap.add_argument("--out", default=None, help="defaults to the config's out_dir")
    ap.add_argument("--n-episodes", type=int, default=None,
                    help="override n_episodes (per policy x persona cell)")
    ap.add_argument("--seed", type=int, default=None, help="override master_seed")
    ap.add_argument("--extractor", default=None,
                    help="override extractor.type; only 'stub' exists (§5)")
    args = ap.parse_args(argv)

    cfg = effective_config(load_config(args.config), args.n_episodes, args.seed)
    if args.extractor:
        cfg["extractor"]["type"] = args.extractor
    if cfg["extractor"]["type"] == "p2":
        from askback.extract.p2_adapter import P2Extractor

        try:
            P2Extractor()
        except NotImplementedError as exc:
            # ASCII: Windows consoles default to cp1252 and mangle the sign.
            raise SystemExit(f"{exc} (PRD section 5)") from exc
    print(f"wrote {run_study(cfg, args.out or cfg['out_dir'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())