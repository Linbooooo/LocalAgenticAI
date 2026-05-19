from __future__ import annotations

import os
import platform
import subprocess


def hardware_report() -> str:
    lines = [
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"CPU cores visible: {os.cpu_count() or 'unknown'}",
    ]
    lines.extend(_memory_lines())
    lines.extend(_gpu_lines())
    return "\n".join(lines)


def _memory_lines() -> list[str]:
    meminfo = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                meminfo[key] = raw_value.strip()
    except OSError:
        return ["Memory: unknown"]

    total = meminfo.get("MemTotal", "unknown")
    available = meminfo.get("MemAvailable", "unknown")
    swap = meminfo.get("SwapTotal", "unknown")
    return [f"Memory total: {total}", f"Memory available: {available}", f"Swap total: {swap}"]


def _gpu_lines() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ["GPU: none detected by nvidia-smi"]

    if completed.returncode != 0 or not completed.stdout.strip():
        return ["GPU: none detected by nvidia-smi"]
    return [f"GPU: {line.strip()}" for line in completed.stdout.splitlines() if line.strip()]

