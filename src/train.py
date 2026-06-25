"""Single entrypoint: scaffold-split KIBA, run the classical baseline sweep and
the GNN, log everything to MLflow (local), and append results to
``results/metrics.csv``.

MLflow run layout (nested):
    parent run  "algorithm_sweep"   <- logs shared params (cv_folds, seed, split)
      |-- child  "LinearRegression"
      |-- child  "Ridge"
      |-- ...                        <- one per classical regressor
      |-- child  "GNN_GraphDTA"

Metrics are REGRESSION metrics (RMSE, MAE, Pearson r, R^2). Classification
metrics (accuracy/precision/recall/F1/ROC-AUC) are intentionally NOT logged —
they are undefined for continuous pKd/affinity prediction.

Run:
    python src/train.py                 # full KIBA run
    python src/train.py --smoke         # tiny toy slice, 1 GNN epoch (CI/dev)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mlflow
import numpy as np

# Allow `python src/train.py` from repo root by making `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affinity_gnn import baseline_sweep, evaluate, features  # noqa: E402
from affinity_gnn.data.loader import load_kiba, load_toy  # noqa: E402
from affinity_gnn.splits import assert_no_scaffold_overlap, scaffold_split  # noqa: E402

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "metrics.csv"
EXPERIMENT = "kiba-binding-affinity"


HEADER = ["model_name", "params", "rmse", "mae", "pearson_r", "r2"]


def append_metrics_row(model_name: str, params: dict, metrics: dict) -> None:
    """Upsert one row into the committed results table, keyed by model_name.

    Re-running a single model (e.g. a longer GNN) replaces its existing row
    rather than appending a duplicate, so the committed table always holds one
    row per model.
    """
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as fh:
            reader = list(csv.reader(fh))
        rows = [r for r in reader[1:] if r and r[0] != model_name]
    rows.append([
        model_name,
        ";".join(f"{k}={v}" for k, v in params.items()),
        round(metrics["rmse"], 4),
        round(metrics["mae"], 4),
        round(metrics["pearson_r"], 4),
        round(metrics["r2"], 4),
    ])
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


def run_baseline_sweep(X_train, y_train, X_test, y_test, cv_folds, seed) -> None:
    """One MLflow child run per classical regressor."""
    for spec in baseline_sweep.build_model_specs(random_seed=seed):
        with mlflow.start_run(run_name=spec.name, nested=True):
            mlflow.log_param("model_name", spec.name)
            mlflow.log_param("featurization", "morgan_r2_1024+aac")
            mlflow.log_param("split_strategy", "scaffold")
            for k, v in spec.params.items():
                mlflow.log_param(k, v)

            metrics = baseline_sweep.fit_and_test_metrics(
                spec.estimator, X_train, y_train, X_test, y_test
            )
            # Regression metrics only (see module docstring for why).
            mlflow.log_metrics(metrics)
            # cloudpickle (not the newer skops default, which rejects
            # third-party estimator types like XGBoost's Booster).
            mlflow.sklearn.log_model(
                spec.estimator, name="model",
                serialization_format="cloudpickle",
            )
            append_metrics_row(spec.name, spec.params, metrics)
            print(f"[baseline] {spec.name}: {metrics}")


def run_gnn(
    train_rows, test_rows, *, hidden_dim, num_layers, lr, epochs, seed, batch_size
) -> None:
    """One MLflow child run for the GNN."""
    import torch
    from torch_geometric.loader import DataLoader

    from affinity_gnn.gnn_model import GraphDTA, predict, train_one_epoch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_graphs = features.build_graph_dataset(
        train_rows["smiles"], train_rows["affinity"], train_rows["sequence"]
    )
    test_graphs = features.build_graph_dataset(
        test_rows["smiles"], test_rows["affinity"], test_rows["sequence"]
    )
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    model = GraphDTA(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name="GNN_GraphDTA", nested=True):
        mlflow.log_params({
            "model_name": "GNN_GraphDTA",
            "featurization": "molecular_graph+cnn_protein",
            "split_strategy": "scaffold",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": lr,
            "epochs": epochs,
            "random_seed": seed,
        })
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            mlflow.log_metric("train_mse", loss, step=epoch)
            print(f"[gnn] epoch {epoch + 1}/{epochs} train_mse={loss:.4f}")

        y_true, y_pred = predict(model, test_loader, device)
        metrics = evaluate.regression_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        # GNN is a torch module — mlflow.sklearn won't accept it. Use pickle
        # serialization (not the newer 'pt2' traced-graph default, which needs
        # an input_example — awkward for a PyG mini-batch).
        mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle",
        )
        # Also persist to a stable path so the model can be reloaded without
        # digging through MLflow run ids. We save the constructor config
        # alongside the weights so the architecture can be rebuilt exactly.
        models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {"hidden_dim": hidden_dim, "num_layers": num_layers},
            },
            models_dir / "gnn_graphdta.pt",
        )
        print(f"[gnn] saved model -> {models_dir / 'gnn_graphdta.pt'}")
        append_metrics_row(
            "GNN_GraphDTA",
            {"hidden_dim": hidden_dim, "num_layers": num_layers, "lr": lr,
             "epochs": epochs},
            metrics,
        )
        print(f"[gnn] test: {metrics}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline sweep + GNN with MLflow tracking.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny toy data, 1 GNN epoch — for CI / quick checks")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="subsample KIBA to N pairs (speed)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gnn-only", action="store_true",
                    help="skip the classical sweep; train/refresh only the GNN")
    args = ap.parse_args()

    # Smoke runs use throwaway toy data — never let them overwrite the committed,
    # reviewer-facing results table.
    if args.smoke:
        global RESULTS_CSV
        RESULTS_CSV = RESULTS_CSV.parent / "metrics_smoke.csv"

    if args.smoke:
        ds = load_toy(n=64, seed=args.seed)
        epochs, batch_size = 1, 8
    else:
        ds = load_kiba(max_pairs=args.max_pairs, seed=args.seed)
        epochs, batch_size = args.epochs, args.batch_size

    frame = ds.frame
    train_idx, _, test_idx = scaffold_split(
        ds.smiles, frac_train=0.8, frac_valid=0.0, frac_test=0.2, seed=args.seed
    )
    assert_no_scaffold_overlap(ds.smiles, train_idx, test_idx)
    print(f"scaffold split: {len(train_idx)} train / {len(test_idx)} test pairs")

    train_rows = frame.iloc[train_idx].reset_index(drop=True)
    test_rows = frame.iloc[test_idx].reset_index(drop=True)

    # Baseline features (skipped entirely in --gnn-only mode to save time)
    if not args.gnn_only:
        X_train = features.featurize_baseline(
            train_rows["smiles"].tolist(), train_rows["sequence"].tolist()
        )
        X_test = features.featurize_baseline(
            test_rows["smiles"].tolist(), test_rows["sequence"].tolist()
        )
        y_train = train_rows["affinity"].to_numpy(np.float32)
        y_test = test_rows["affinity"].to_numpy(np.float32)

    run_name = "gnn_only" if args.gnn_only else "algorithm_sweep"
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        # Shared config logged once on the parent.
        mlflow.log_params({
            "cv_folds": args.cv_folds,
            "random_seed": args.seed,
            "split_strategy": "scaffold",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "dataset": "KIBA" if not args.smoke else "toy",
        })
        if not args.gnn_only:
            run_baseline_sweep(X_train, y_train, X_test, y_test,
                               args.cv_folds, args.seed)
        run_gnn(
            train_rows, test_rows,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, lr=args.lr,
            epochs=epochs, seed=args.seed, batch_size=batch_size,
        )

    print(f"\nDone. Results appended to {RESULTS_CSV}")
    print("Inspect runs with:  mlflow ui   (http://127.0.0.1:5000)")


if __name__ == "__main__":
    main()
