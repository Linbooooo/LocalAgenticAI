from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import LocalAgent
from .config import AgentConfig
from .hardware import hardware_report
from .ollama_client import OllamaClient, OllamaConnectionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-agent",
        description="Run a local-only agentic coding assistant backed by Ollama.",
    )
    parser.add_argument("command", nargs="*", help="Task text, or one of: chat, doctor, hardware, preload")
    parser.add_argument("--config", type=Path, help="Path to a JSON config override file.")
    parser.add_argument("--model", help="Override the model name.")
    parser.add_argument("--workspace", type=Path, help="Workspace the agent may read and edit.")
    parser.add_argument("--yes", action="store_true", help="Allow mutating tools without interactive prompts.")
    parser.add_argument("--max-steps", type=int, help="Maximum agent action steps before stopping.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig.load(args.config)

    if args.model:
        config.model = args.model
    if args.workspace:
        config.workspace = args.workspace
    if args.yes:
        config.trust = "auto"
    if args.max_steps:
        config.max_steps = args.max_steps
    config.finalize()

    command = args.command[0] if args.command else "chat"
    rest = args.command[1:]

    if command == "doctor":
        return doctor(config)
    if command == "hardware":
        print(hardware_report())
        return 0
    if command == "preload":
        return preload(config)
    if command == "chat":
        return chat(config)

    task = " ".join(args.command).strip()
    if not task:
        return chat(config)
    return run_task(config, task)


def doctor(config: AgentConfig) -> int:
    print("Local Agentic AI doctor")
    print()
    print(hardware_report())
    print()
    print(f"Workspace: {config.workspace}")
    print(f"Model: {config.model}")
    print(f"Ollama URL: {config.ollama_url}")
    print(f"Context: {config.num_ctx}")
    print(f"Context cap: {config.max_num_ctx}")
    print(f"Trust: {config.trust}")
    print()

    client = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
    try:
        version = client.version()
        print(f"Ollama: reachable ({version})")
    except OllamaConnectionError as exc:
        print(f"Ollama: not reachable ({exc})")
        print("Start it with: bash scripts/run_ollama_tuned.sh")
        return 1

    try:
        running = client.ps().get("models", [])
    except OllamaConnectionError:
        running = []
    if running:
        print("Loaded models:")
        for model in running:
            name = model.get("name", "unknown")
            context = model.get("context_length", "unknown")
            size = model.get("size", "unknown")
            vram = model.get("size_vram", "unknown")
            print(f"  {name}: context={context}, size={size}, vram={vram}")
            if vram == 0:
                print("  warning: Ollama reports 0 bytes in VRAM; inference may be CPU-bound.")
    else:
        print("Loaded models: none")

    models = client.tags()
    names = {model.get("name") for model in models.get("models", [])}
    if config.model in names:
        print(f"Model installed: yes ({config.model})")
    else:
        print(f"Model installed: no ({config.model})")
        print(f"Pull it with: ollama pull {config.model}")
        return 1
    return 0


def preload(config: AgentConfig) -> int:
    client = OllamaClient(config.ollama_url, timeout=config.ollama_timeout)
    try:
        client.chat(
            model=config.model,
            messages=[{"role": "user", "content": ""}],
            options=config.ollama_options(),
            keep_alive=-1,
        )
    except OllamaConnectionError as exc:
        print(f"Could not preload model: {exc}", file=sys.stderr)
        return 1
    print(f"Preloaded {config.model}")
    return 0


def chat(config: AgentConfig) -> int:
    print("Local Agentic AI. Type /exit to stop.")
    agent = LocalAgent(config)
    while True:
        try:
            task = input("> ").strip()
        except EOFError:
            print()
            return 0
        if task in {"/exit", "/quit"}:
            return 0
        if not task:
            continue
        result = agent.run(task)
        print(result.content)


def run_task(config: AgentConfig, task: str) -> int:
    agent = LocalAgent(config)
    try:
        result = agent.run(task)
    except OllamaConnectionError as exc:
        print(f"Ollama is not reachable: {exc}", file=sys.stderr)
        print("Start it with: bash scripts/run_ollama_tuned.sh", file=sys.stderr)
        return 1
    print(result.content)
    return 0
