#!/usr/bin/env python3
"""Pilot run: all 6 conditions on 5 tasks x 1 trial (30 sims).

Validates that all conditions complete without crashes and results are plausible.
"""

import json
import os
import subprocess
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "experiment.yaml")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run.py")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    task_ids = config["pilot_task_ids"]
    agent_llm_args = config.get("agent_llm_args", {})

    for name, condition in config["conditions"].items():
        print(f"\n{'='*60}")
        print(f"PILOT: {name}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, RUN_SCRIPT,
            "run",
            "--domain", config["domain"],
            "--agent", condition["agent"],
            "--agent-llm", config["agent_llm"],
            "--user-llm", config["user_llm"],
            "--retrieval-config", condition["retrieval_config"],
            "--num-trials", "1",
            "--max-concurrency", str(config["max_concurrency"]),
            "--task-ids"] + task_ids + [
            "--seed", str(config["seed"]),
            "--verbose-logs",
            "--save-to", f"pilot_{name}",
        ]

        if agent_llm_args:
            cmd.extend(["--agent-llm-args", json.dumps(agent_llm_args)])

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "tau2-bench"))

        if result.returncode != 0:
            print(f"\nERROR: {name} failed with return code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("PILOT COMPLETE: All 6 conditions finished successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
