"""Run the selected model per circuit for the current experiment.

Bahrain, Saudi, USA  -> LR-EW + XGB-EW  (best window ratios encoded in each YAML)
Hungary, Italy       -> LR-EW + XGB-SW  (best window ratios encoded in each YAML)

Usage:
    python Scripts/Source/run_experiment.py
    python Scripts/Source/run_experiment.py --continue-on-error
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from modeling_utils import load_simple_yaml


EXPERIMENT_PLAN = [
    ("bahrain.yaml",  ["model_lr_ew.py", "model_xgb_ew.py"]),
    ("saudi.yaml",    ["model_lr_ew.py", "model_xgb_ew.py"]),
    ("usa.yaml",      ["model_lr_ew.py", "model_xgb_ew.py"]),
    ("hungary.yaml",  ["model_lr_ew.py", "model_xgb_sw.py"]),
    ("italy.yaml",    ["model_lr_ew.py", "model_xgb_ew.py"]),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining jobs when one step fails.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    args = parse_args()
    root = repo_root()
    script_dir = Path(__file__).resolve().parent
    configs_dir = root / "configs"

    failures = []
    for config_name, scripts in EXPERIMENT_PLAN:
        config_path = configs_dir / config_name
        if not config_path.exists():
            print(f"Config not found, skipping: {config_path}", file=sys.stderr)
            continue

        config = load_simple_yaml(config_path)
        target_gp_name = str(config.get("target_gp_name", config_path.stem))

        for script_name in scripts:
            script_path = script_dir / script_name
            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["TARGET_GP_NAME"] = target_gp_name
            env["MLFLOW_ALLOW_FILE_STORE"] = "true"

            print("\n" + "=" * 80, flush=True)
            print(f"Grand Prix : {target_gp_name}", flush=True)
            print(f"Script     : {script_name}", flush=True)
            print(f"Config     : {config_path}", flush=True)
            print("=" * 80, flush=True)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=root,
                env=env,
                check=False,
            )
            if result.returncode != 0:
                failures.append((config_name, script_name, result.returncode))
                print(
                    f"FAILED: {script_name} / {config_name} (exit {result.returncode})",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    return result.returncode

    if failures:
        print("\nCompleted with failures:")
        for cfg, scr, code in failures:
            print(f"  {cfg} | {scr} | exit {code}")
        return 1

    print("\nAll experiment runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
