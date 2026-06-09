from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_agent.agent import LocalAgent
from local_agent.config import AgentConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench Lite patches with the local agent and optionally run the official evaluator."
    )
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--workdir", type=Path, default=Path(".swebench-work"))
    parser.add_argument("--output", type=Path, default=Path("swebench_predictions.jsonl"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--evaluate", action="store_true", help="Run the official Docker evaluator after generation.")
    parser.add_argument("--evaluate-only", action="store_true", help="Score an existing predictions file.")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    if args.evaluate_only:
        if not args.output.exists():
            raise SystemExit(f"Predictions file does not exist: {args.output}")
        return run_official_evaluator(args)

    instances = load_instances(args.dataset_name, args.split, args.instance_id, args.limit)
    args.workdir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    trajectories = args.output.with_suffix(".trajectories.jsonl")

    with args.output.open("w", encoding="utf-8") as prediction_file, trajectories.open("w", encoding="utf-8") as trace_file:
        for instance in instances:
            prediction, trace = run_instance(instance, args)
            prediction_file.write(json.dumps(prediction) + "\n")
            prediction_file.flush()
            trace_file.write(json.dumps(trace) + "\n")
            trace_file.flush()
            predictions.append(prediction)
            print(f"{instance['instance_id']}: patch bytes={len(prediction['model_patch'])}")

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Wrote trajectories to {trajectories}")
    if args.evaluate:
        return run_official_evaluator(args)
    print("Run again with --evaluate after installing the official `swebench` package and Docker.")
    return 0


def load_instances(
    dataset_name: str,
    split: str,
    instance_ids: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the benchmark extra first: python3 -m pip install datasets swebench") from exc

    dataset = load_dataset(dataset_name, split=split)
    selected = []
    wanted = set(instance_ids or [])
    for row in dataset:
        instance = dict(row)
        if wanted and instance["instance_id"] not in wanted:
            continue
        selected.append(instance)
        if len(selected) >= limit:
            break
    if not selected:
        raise SystemExit("No matching SWE-bench instances found.")
    return selected


def run_instance(instance: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    instance_id = str(instance["instance_id"])
    workspace = args.workdir / instance_id
    if workspace.exists():
        raise SystemExit(f"Workspace already exists; move or remove it before rerunning: {workspace}")

    clone_url = f"https://github.com/{instance['repo']}.git"
    subprocess.run(["git", "clone", "--quiet", clone_url, str(workspace)], check=True)
    subprocess.run(["git", "checkout", "--quiet", str(instance["base_commit"])], cwd=workspace, check=True)

    config = AgentConfig(
        model=args.model,
        ollama_url=args.url,
        workspace=workspace,
        trust="auto",
        max_steps=args.max_steps,
        shell_timeout=300,
    )
    config.finalize()
    agent = LocalAgent(config)
    prompt = (
        "Resolve this GitHub issue in the checked-out repository. Inspect the code, make the smallest correct patch, "
        "and run relevant tests when possible. Stay on the checked-out base commit; do not switch branches or commits. "
        "Do not use the network.\n\n"
        f"{instance['problem_statement']}"
    )
    result = agent.run(prompt)
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": args.model,
        "model_patch": patch,
    }
    trace = {
        "instance_id": instance_id,
        "answer": result.content,
        "turns": result.turns,
        "commands": result.commands,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "tfs_ms": None if result.time_to_first_shell_ms is None else round(result.time_to_first_shell_ms, 2),
        "model_calls": [metric.__dict__ for metric in result.model_metrics],
        "base_commit_preserved": current_commit == str(instance["base_commit"]),
        "messages": agent.messages,
    }
    if current_commit != str(instance["base_commit"]):
        prediction["model_patch"] = ""
        trace["benchmark_error"] = f"Agent changed HEAD from {instance['base_commit']} to {current_commit}."
    return prediction, trace


def run_official_evaluator(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--predictions_path",
        str(args.output),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        "local-agent",
    ]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
