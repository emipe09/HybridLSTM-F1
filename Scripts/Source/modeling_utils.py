"""Shared utilities for the sliding-window and expanding-window modeling scripts."""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import scipy.stats as stats
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments.
    stats = None


CONFIG_ALIASES = {
    "bahrain_grand_prix": "bahrain.yaml",
    "saudi_arabian_grand_prix": "saudi.yaml",
    "united_states_grand_prix": "usa.yaml",
    "italian_grand_prix": "italy.yaml",
    "hungarian_grand_prix": "hungary.yaml",
}

REQUIRED_CONFIG_KEYS = [
    "target_gp_name",
    "model_data_dir",
    "results_dir",
    "cleaned_data_filename_template",
    "xgb_params_subdir",
    "xgb_params_filename_template",
    "xgb_models_subdir",
    "xgb_model_filename_template",
    "xgb_model_metadata_filename_template",
    "lr_models_subdir",
    "lr_model_filename_template",
    "lr_model_metadata_filename_template",
    "target_col",
    "lap_col",
    "numerical_features",
    "categorical_features",
    "holdout_ratio",
    "window_ratio",
    "window_train_ratio",
    "window_step_ratio",
    "alpha_cos",
    "beta_cos",
    "random_seed",
    "optuna_trials",
    "use_saved_xgb_params",
    "use_saved_lstm_params",
    "mlflow_enabled",
    "mlflow_tracking_uri",
    "mlflow_experiment_name",
]


def safe_gp_name(gp_name: str) -> str:
    return gp_name.lower().replace(" ", "_")


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        raw_items = value[1:-1].split(",")
        return [parse_scalar(item.strip()) for item in raw_items if item.strip()]
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict:
    config = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = parse_scalar(value)
    return config


def validate_config(config: dict, config_path: Path) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        joined_keys = ", ".join(missing)
        raise KeyError(f"Missing required config key(s) in {config_path}: {joined_keys}")


def load_config(repo_root: Path) -> tuple[dict, Path | None]:
    env_target = os.environ.get("TARGET_GP_NAME")
    env_config_path = os.environ.get("CONFIG_PATH")
    if env_config_path:
        config_path = Path(env_config_path)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
    else:
        config_dir = repo_root / "configs"
        if not env_target:
            raise ValueError("Set CONFIG_PATH or TARGET_GP_NAME so the script can load the circuit YAML.")
        safe_name = safe_gp_name(env_target)
        aliased_name = CONFIG_ALIASES.get(safe_name, f"{safe_name}.yaml")
        config_path = config_dir / aliased_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_simple_yaml(config_path)
    validate_config(config, config_path)

    if env_target and not env_config_path and str(config["target_gp_name"]) != env_target:
        raise ValueError(
            f"TARGET_GP_NAME={env_target!r} resolved to {config_path}, "
            f"but the YAML target_gp_name is {config['target_gp_name']!r}."
        )

    return config, config_path


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def build_cleaned_data_path(repo_root: Path, config: dict) -> Path:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    filename_template = str(config["cleaned_data_filename_template"])
    filename = filename_template.format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    return resolve_repo_path(repo_root, str(config["model_data_dir"])) / target_gp_name / filename


def build_xgb_params_path(repo_root: Path, config: dict) -> Path:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    filename_template = str(config["xgb_params_filename_template"])
    filename = filename_template.format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    return (
        resolve_repo_path(repo_root, str(config["results_dir"]))
        / str(config["xgb_params_subdir"])
        / filename
    )


def build_xgb_model_paths(repo_root: Path, config: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(config["xgb_model_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    metadata_filename = str(config["xgb_model_metadata_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / str(config["xgb_models_subdir"])
    return model_dir / model_filename, model_dir / metadata_filename


def build_lr_model_paths(repo_root: Path, config: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(config["lr_model_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    metadata_filename = str(config["lr_model_metadata_filename_template"]).format(
        target_gp_name=target_gp_name,
        safe_gp_name=safe_name,
    )
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / str(config["lr_models_subdir"])
    return model_dir / model_filename, model_dir / metadata_filename


def build_xgb_ew_params_path(repo_root: Path, config: dict) -> Path:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    filename_template = str(
        config.get("ew_xgb_params_filename_template", "{safe_gp_name}_xgb_params_ew.json")
    )
    filename = filename_template.format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    subdir = str(config.get("ew_xgb_params_subdir", "xgboost/ew/params"))
    return resolve_repo_path(repo_root, str(config["results_dir"])) / subdir / filename


def build_xgb_ew_model_paths(repo_root: Path, config: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(
        config.get("ew_xgb_model_filename_template", "{safe_gp_name}_xgb_model_ew.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    metadata_filename = str(
        config.get("ew_xgb_model_metadata_filename_template", "{safe_gp_name}_xgb_model_ew_metadata.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    subdir = str(config.get("ew_xgb_models_subdir", "xgboost/ew/models"))
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return model_dir / model_filename, model_dir / metadata_filename


def build_lr_ew_model_paths(repo_root: Path, config: dict) -> tuple[Path, Path]:
    target_gp_name = str(config["target_gp_name"])
    safe_name = safe_gp_name(target_gp_name)
    model_filename = str(
        config.get("ew_lr_model_filename_template", "{safe_gp_name}_lr_model_ew.pkl")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    metadata_filename = str(
        config.get("ew_lr_model_metadata_filename_template", "{safe_gp_name}_lr_model_ew_metadata.json")
    ).format(target_gp_name=target_gp_name, safe_gp_name=safe_name)
    subdir = str(config.get("ew_lr_models_subdir", "linear_regression/ew/models"))
    model_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / subdir
    return model_dir / model_filename, model_dir / metadata_filename


def resolve_mlflow_tracking_uri(repo_root: Path, config: dict) -> str:
    tracking_uri = str(config.get("mlflow_tracking_uri", "Scripts/Results/mlruns"))
    if "://" in tracking_uri or tracking_uri.startswith("databricks"):
        return tracking_uri
    return resolve_repo_path(repo_root, tracking_uri).resolve().as_uri()


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def flatten_metrics(metrics: dict, prefix: str = "") -> dict[str, float]:
    flattened = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=f"{metric_name}_"))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            flattened[f"{metric_name}_lower"] = float(value[0])
            flattened[f"{metric_name}_upper"] = float(value[1])
        elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
            flattened[metric_name] = float(value)
    return flattened


def log_mlflow_run(
    repo_root: Path,
    config: dict,
    model_name: str,
    num_cols: list[str],
    cat_cols: list[str],
    split_info: dict,
    window_results: dict,
    summary_metrics: dict,
    extra_params: dict | None = None,
    artifacts: list[Path] | None = None,
    validation_mode: str = "sw",
):
    if not bool(config.get("mlflow_enabled", True)):
        return None

    try:
        import mlflow
    except ModuleNotFoundError:
        print("MLflow tracking skipped: install mlflow with `pip install -r Utils/requirements.txt`.")
        return None

    safe_name = safe_gp_name(str(config["target_gp_name"]))
    tracking_uri = resolve_mlflow_tracking_uri(repo_root, config)
    experiment_name = str(config.get("mlflow_experiment_name", "f1-lap-time-{safe_gp_name}")).format(
        target_gp_name=config["target_gp_name"],
        safe_gp_name=safe_name,
    )
    run_name = f"{safe_name}-{model_name}-{validation_mode}"

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = {
        "model_name": model_name,
        "target_gp_name": config["target_gp_name"],
        "target_col": config["target_col"],
        "lap_col": config["lap_col"],
        "numerical_features": ", ".join(num_cols),
        "categorical_features": ", ".join(cat_cols),
        "holdout_ratio": config["holdout_ratio"],
        "window_ratio": config["window_ratio"],
        "window_train_ratio": config["window_train_ratio"],
        "window_step_ratio": config["window_step_ratio"],
        "alpha_cos": config["alpha_cos"],
        "beta_cos": config["beta_cos"],
        "random_seed": config["random_seed"],
        **split_info,
        **(extra_params or {}),
    }

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({key: json_ready(value) for key, value in params.items()})
        mlflow.log_metrics(flatten_metrics(summary_metrics))

        for index, window_id in enumerate(window_results.get("window", [])):
            step = int(window_id)
            for metric_name in ("rmse", "mae", "r2", "std"):
                metric_values = window_results.get(metric_name, [])
                if index < len(metric_values):
                    mlflow.log_metric(f"{validation_mode}_window_{metric_name}", float(metric_values[index]), step=step)

        temp_artifact_dir = resolve_repo_path(repo_root, str(config["results_dir"])) / "mlflow_tmp" / active_run.info.run_id
        temp_artifact_dir.mkdir(parents=True, exist_ok=True)
        generated_artifacts = {
            "config.json": config,
            f"{validation_mode}_window_results.json": window_results,
            "summary_metrics.json": summary_metrics,
        }
        for artifact_name, artifact_data in generated_artifacts.items():
            artifact_file = temp_artifact_dir / artifact_name
            artifact_file.write_text(json.dumps(json_ready(artifact_data), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact_file))

        for artifact_path in artifacts or []:
            artifact_path = Path(artifact_path)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        print(f"MLflow run logged: {active_run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")
        return active_run.info.run_id


def load_cleaned_data(script_path: Path) -> tuple[str, dict, Path, pd.DataFrame]:
    script_dir = script_path.resolve().parent
    scripts_dir = script_dir.parent
    repo_root = scripts_dir.parent
    config, config_path = load_config(repo_root)
    target_gp_name = str(config["target_gp_name"])
    input_csv_path = build_cleaned_data_path(repo_root, config)

    print(f"Using config:\n{config_path}")
    print(f"Loading cleaned data from:\n{input_csv_path}")
    if not input_csv_path.exists():
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    return target_gp_name, config, repo_root, pd.read_csv(input_csv_path)


def select_modeling_columns(df_base: pd.DataFrame, config: dict):
    numerical_features = list(config["numerical_features"])
    categorical_features = list(config["categorical_features"])

    def ordered_existing_unique(columns):
        selected = []
        seen = set()
        for col in columns:
            if col in seen or col not in df_base.columns:
                continue
            selected.append(col)
            seen.add(col)
        return selected

    num_cols = ordered_existing_unique(numerical_features)
    cat_cols = ordered_existing_unique(categorical_features)
    return num_cols, cat_cols


def prepare_raw_features(df_base: pd.DataFrame, num_cols: list[str], cat_cols: list[str], target_col: str):
    X_raw = df_base[num_cols + cat_cols].copy()
    y_raw = df_base[target_col].copy()
    valid_indices = y_raw.dropna().index
    return X_raw.loc[valid_indices], y_raw.loc[valid_indices], valid_indices


def align_one_hot(X_train, X_eval, cat_cols, drop_first):
    X_train = X_train.copy()
    X_eval = X_eval.copy()

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_eval[col] = X_eval[col].fillna("Missing").astype(str)

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=drop_first)
    X_eval_enc = pd.get_dummies(X_eval, columns=cat_cols, drop_first=drop_first)
    X_eval_enc = X_eval_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    return X_train_enc, X_eval_enc


def fit_predict_linear_regression(X_train, y_train, X_eval, cat_cols):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X_train_enc, X_eval_enc = align_one_hot(X_train, X_eval, cat_cols, drop_first=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    model = LinearRegression()
    X_train_imp = imputer.fit_transform(X_train_enc)
    X_eval_imp = imputer.transform(X_eval_enc)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_eval_scaled = scaler.transform(X_eval_imp)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_eval_scaled)
    return preds, model, imputer, scaler, X_train_enc.columns


def calc_stats(values):
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))
    if len(values) > 1 and stats is not None:
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean_value, scale=stats.sem(values))
    elif len(values) > 1:
        margin = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
        ci = (mean_value - margin, mean_value + margin)
    else:
        ci = (mean_value, mean_value)
    return mean_value, float(ci[0]), float(ci[1])


def calc_holdout_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rmse_point = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_point = float(mean_absolute_error(y_true, y_pred))
    r2_point = float(r2_score(y_true, y_pred))

    if n < 2:
        return {"rmse": (rmse_point, rmse_point), "mae": (mae_point, mae_point), "r2": (r2_point, r2_point)}

    rng = np.random.default_rng(seed)
    rmse_samples, mae_samples, r2_samples = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_pred[idx]
        rmse_samples.append(np.sqrt(mean_squared_error(yb, pb)))
        mae_samples.append(mean_absolute_error(yb, pb))
        r2_value = r2_score(yb, pb)
        if np.isfinite(r2_value):
            r2_samples.append(r2_value)

    def percentile_ci(samples, point_value):
        if not samples:
            return point_value, point_value
        lower = float(np.percentile(samples, 100 * (alpha / 2)))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return lower, upper

    return {
        "rmse": percentile_ci(rmse_samples, rmse_point),
        "mae": percentile_ci(mae_samples, mae_point),
        "r2": percentile_ci(r2_samples, r2_point),
    }


def calc_cos_metric(error_sliding, error_final, std_sliding, std_final, alpha=0.5, beta=0.5):
    error_sliding = float(error_sliding)
    error_final = float(error_final)
    std_sliding = float(std_sliding)
    std_final = float(std_final)

    if np.isclose(error_sliding, 0) or np.isclose(std_sliding, 0):
        return np.nan, error_sliding, error_final, std_sliding, std_final

    cos_value = alpha * (error_final / error_sliding) + beta * (std_final / std_sliding)
    return cos_value, error_sliding, error_final, std_sliding, std_final


def build_sliding_windows(n_laps, window_ratio, train_ratio, step_ratio):
    if n_laps < 2:
        raise ValueError("Insufficient data for sliding window validation.")

    window_size = max(2, min(int(np.ceil(n_laps * window_ratio)), n_laps))
    train_size = max(1, int(np.floor(window_size * train_ratio)))
    if train_size >= window_size:
        train_size = window_size - 1

    val_size = window_size - train_size
    step_size = max(1, int(np.ceil(window_size * step_ratio)))

    windows = []
    start = 0
    while start + window_size <= n_laps:
        windows.append((start, start + train_size, start + window_size))
        start += step_size

    last_start = n_laps - window_size
    if not windows or windows[-1][0] != last_start:
        windows.append((last_start, last_start + train_size, last_start + window_size))

    return windows, window_size, train_size, val_size, step_size


def build_expanding_windows(n_laps, window_ratio, train_ratio, step_ratio):
    """
    Generates expanding-window folds within the modeling block.

    The initial training set covers the same lap count as the first SW window's train portion
    (window_ratio * train_ratio * n_laps). The validation chunk equals the SW validation
    portion (window_ratio * (1 - train_ratio) * n_laps). Each fold expands training by one
    val_chunk then validates on the next chunk. Training always starts from lap index 0.
    The step_ratio parameter is accepted for API parity with build_sliding_windows but is
    unused — EW always advances by exactly one val_chunk.
    """
    if n_laps < 2:
        raise ValueError("Insufficient data for expanding window validation.")

    window_size = max(2, min(int(np.ceil(n_laps * window_ratio)), n_laps))
    train_size = max(1, int(np.floor(window_size * train_ratio)))
    if train_size >= window_size:
        train_size = window_size - 1

    val_size = window_size - train_size
    step_size = val_size

    windows = []
    train_end = train_size
    while train_end + val_size <= n_laps:
        windows.append((0, train_end, train_end + val_size))
        train_end += val_size

    if not windows:
        raise ValueError(
            f"No expanding windows could be constructed: n_laps={n_laps}, "
            f"window_ratio={window_ratio}, train_ratio={train_ratio}."
        )

    return windows, window_size, train_size, val_size, step_size


def decode_step_key(key):
    """Decode a composite (Year*10000 + LapNumber) key to a human-readable string."""
    k = int(key)
    return f"lap {k % 10000} year {k // 10000}"


def build_sequential_split(df_base, valid_indices, holdout_ratio, lap_col):
    """Split data into modeling and holdout blocks using temporal ordering.

    If a 'Year' column exists the split is performed on unique (Year, LapNumber)
    pairs ordered chronologically, so the holdout contains the most recent laps
    across years rather than the highest lap numbers.  When no 'Year' column is
    present the function falls back to LapNumber-only ordering.

    The returned step_series contains composite keys (Year*10000 + LapNumber)
    when Year is available, or raw LapNumber values otherwise.  All downstream
    code should use step_series for sorting and masking rather than raw LapNumber.
    """
    if lap_col not in df_base.columns:
        raise KeyError(f"Column '{lap_col}' not found.")

    lap_raw = df_base.loc[valid_indices, lap_col]
    if lap_raw.dropna().empty:
        raise ValueError("No valid lap values are available.")

    if "Year" in df_base.columns:
        year_raw = df_base.loc[valid_indices, "Year"].astype(int)
        step_series = year_raw * 10000 + lap_raw.astype(int)
    else:
        step_series = lap_raw.copy()

    unique_steps = np.sort(pd.to_numeric(step_series, errors="coerce").dropna().unique())
    total_laps = len(unique_steps)

    holdout_count = max(1, int(np.ceil(total_laps * holdout_ratio)))
    holdout_count = min(holdout_count, total_laps - 1)
    model_steps = unique_steps[: total_laps - holdout_count]
    holdout_steps = unique_steps[total_laps - holdout_count :]

    model_mask = step_series.isin(model_steps)
    holdout_mask = step_series.isin(holdout_steps)
    model_idx = step_series[model_mask].index
    holdout_idx = step_series[holdout_mask].index

    if len(model_idx) == 0 or len(holdout_idx) == 0:
        raise ValueError("Invalid sequential split: modeling or holdout block is empty.")

    step_min = int(unique_steps[0])
    step_max = int(unique_steps[-1])
    holdout_start_step = int(holdout_steps[0])
    model_end_step = int(model_steps[-1])

    return (
        step_series,
        step_min,
        step_max,
        model_idx,
        holdout_idx,
        holdout_start_step,
        model_end_step,
        total_laps,
    )


def summarize_cos(results, mae_m, rmse_m, mae_holdout, rmse_holdout, std_m, std_holdout, alpha_cos, beta_cos):
    cos_mae, mae_sw, mae_final, std_sw, std_final = calc_cos_metric(
        mae_m, mae_holdout, std_m, std_holdout, alpha=alpha_cos, beta=beta_cos
    )
    cos_rmse, rmse_sw, rmse_final, _, _ = calc_cos_metric(
        rmse_m, rmse_holdout, std_m, std_holdout, alpha=alpha_cos, beta=beta_cos
    )

    cos_mae_windows = alpha_cos * (mae_holdout / np.array(results["mae"])) + beta_cos * (
        std_holdout / np.array(results["std"])
    )
    cos_rmse_windows = alpha_cos * (rmse_holdout / np.array(results["rmse"])) + beta_cos * (
        std_holdout / np.array(results["std"])
    )
    _, cos_mae_l, cos_mae_u = calc_stats(cos_mae_windows)
    _, cos_rmse_l, cos_rmse_u = calc_stats(cos_rmse_windows)

    return {
        "cos_mae": cos_mae,
        "cos_rmse": cos_rmse,
        "cos_mae_ci": (cos_mae_l, cos_mae_u),
        "cos_rmse_ci": (cos_rmse_l, cos_rmse_u),
        "mae_sw": mae_sw,
        "mae_final": mae_final,
        "rmse_sw": rmse_sw,
        "rmse_final": rmse_final,
        "std_sw": std_sw,
        "std_final": std_final,
    }
