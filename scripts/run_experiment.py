#!/usr/bin/env python3
"""Full experiment: 97 tasks x 6 conditions x 3 trials = 1,746 sims.

Runs conditions sequentially at concurrency=3. Designed for overnight runs.
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

    agent_llm_args = config.get("agent_llm_args", {})
    results = {}

    for name, condition in config["conditions"].items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, RUN_SCRIPT,
            "run",
            "--domain", config["domain"],
            "--agent", condition["agent"],
            "--agent-llm", config["agent_llm"],
            "--user-llm", config["user_llm"],
            "--retrieval-config", condition["retrieval_config"],
            "--num-trials", str(config["num_trials"]),
            "--max-concurrency", str(config["max_concurrency"]),
            "--seed", str(config["seed"]),
            "--verbose-logs",
            "--save-to", name,
        ]

        if agent_llm_args:
            cmd.extend(["--agent-llm-args", json.dumps(agent_llm_args)])

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "tau2-bench"))

        if result.returncode == 0:
            results[name] = "OK"
        else:
            results[name] = f"FAILED (exit {result.returncode})"
            print(f"\nERROR: {name} failed with return code {result.returncode}")
            print("Continuing to next condition...")

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker} {name}: {status}")

    failed = [n for n, s in results.items() if s != "OK"]
    if failed:
        print(f"\n{len(failed)} condition(s) failed: {', '.join(failed)}")
        print("Re-run failed conditions with the same --save-to to resume.")
    else:
        print("\nAll conditions completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
