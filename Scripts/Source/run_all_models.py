"""Run all modeling scripts for every configured Grand Prix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from modeling_utils import load_simple_yaml


DEFAULT_CONFIG_ORDER = ["bahrain.yaml", "saudi.yaml", "usa.yaml", "italy.yaml", "hungary.yaml"]

MODEL_SCRIPTS = {
    "lr": "model_lr_sw.py",
    "xgb": "model_xgb_sw.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Linear Regression and XGBoost for all configured Formula 1 Grand Prix datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SCRIPTS.keys(),
        default=list(MODEL_SCRIPTS.keys()),
        help="Model scripts to run. Default: lr xgb.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIG_ORDER,
        help="Config filenames or paths to run. Default: all supported Grand Prix YAML files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining jobs when one model/config pair fails.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config_path(repo_root: Path, config_value: str) -> Path:
    config_path = Path(config_value)
    if not config_path.is_absolute():
        if len(config_path.parts) == 1:
            config_path = repo_root / "configs" / config_path
        else:
            config_path = repo_root / config_path
    return config_path


def run_model(repo_root: Path, script_dir: Path, config_path: Path, model_key: str):
    script_path = script_dir / MODEL_SCRIPTS[model_key]
    config = load_simple_yaml(config_path)
    target_gp_name = str(config.get("target_gp_name", config_path.stem))

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config_path)
    env["TARGET_GP_NAME"] = target_gp_name

    print("\n" + "=" * 80, flush=True)
    print(f"Grand Prix: {target_gp_name}", flush=True)
    print(f"Model: {model_key}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print("=" * 80, flush=True)

    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        env=env,
        check=False,
    )


def main():
    args = parse_args()
    repo_root = repo_root_from_script()
    script_dir = Path(__file__).resolve().parent

    failures = []
    for config_value in args.configs:
        config_path = resolve_config_path(repo_root, config_value)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        for model_key in args.models:
            result = run_model(repo_root, script_dir, config_path, model_key)
            if result.returncode != 0:
                failures.append((config_path, model_key, result.returncode))
                print(
                    f"FAILED: {model_key} with {config_path.name} returned exit code {result.returncode}.",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    return result.returncode

    if failures:
        print("\nCompleted with failures:")
        for config_path, model_key, returncode in failures:
            print(f"- {config_path.name} | {model_key} | exit code {returncode}")
        return 1

    print("\nAll requested model runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
