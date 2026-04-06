from __future__ import annotations

import argparse
import copy

from cytof_archetypes.config import load_config
from cytof_archetypes.experiments import load_suite_config, run_experiment_suite
from cytof_archetypes.notebook_generation import generate_notebooks
from cytof_archetypes.training import evaluate_run_dir, train_from_config


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train probabilistic archetypal autoencoder")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = train_from_config(config)
    print(run_dir)


def evaluate_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing run directory")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument(
        "--checkpoint",
        default="best_checkpoint.pt",
        help="Checkpoint file name in run directory",
    )
    args = parser.parse_args()
    run_dir = evaluate_run_dir(args.run_dir, checkpoint=args.checkpoint)
    print(run_dir)


def generate_notebooks_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis notebooks")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    created = generate_notebooks(args.output_dir)
    for path in created:
        print(path)


def demo_cli() -> None:
    parser = argparse.ArgumentParser(description="Run compact Levine32 end-to-end demo")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-epochs", type=int, default=10, help="Epoch cap for demo")
    args = parser.parse_args()

    config = load_config(args.config)
    config = copy.deepcopy(config)
    config["training"]["max_epochs"] = args.max_epochs
    config["output"]["run_name"] = config["output"].get("run_name") or "demo_run"
    run_dir = train_from_config(config)
    created = generate_notebooks(run_dir / "notebooks_generated")
    print(run_dir)
    for path in created:
        print(path)


def run_suite_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the full probabilistic archetypal deconvolution experiment suite")
    parser.add_argument("--config", required=False, help="Path to suite YAML config")
    args = parser.parse_args()

    config = load_suite_config(args.config)
    out_dir = run_experiment_suite(config)
    print(out_dir)
