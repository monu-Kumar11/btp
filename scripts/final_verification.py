#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master Final Verification Orchestrator (Script 5)

Executes all 4 verification scripts in sequence:
1. scripts/run_final_check.py
2. scripts/check_logs.py
3. scripts/compare_methods.py
4. scripts/test_claim_critic.py

Prints a consolidated summary report and outputs "READY FOR FULL EVALUATION RUN" on clean pass.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Master Verification Suite Orchestrator")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend for verification suite")
    return parser.parse_args()

def run_step(step_name, cmd_args):
    print(f"\n==================================================")
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(cmd_args)}")
    print(f"==================================================")
    try:
        res = subprocess.run(cmd_args, check=True)
        return "PASS", "Completed successfully."
    except subprocess.CalledProcessError as e:
        return "FAIL", f"Subprocess exited with code {e.returncode}."
    except Exception as e:
        return "FAIL", str(e)

def main():
    args = parse_args()
    python_exe = sys.executable

    steps = [
        ("1. Pipeline Execution Check (run_final_check.py)", [python_exe, "scripts/run_final_check.py", "--backend", args.backend]),
        ("2. CSV Log Schema & Metric Validation (check_logs.py)", [python_exe, "scripts/check_logs.py"]),
        ("3. Method Comparison & Hallucination Rate Benchmark (compare_methods.py)", [python_exe, "scripts/compare_methods.py", "--backend", args.backend]),
        ("4. Claim Critic Unit Testing (test_claim_critic.py)", [python_exe, "scripts/test_claim_critic.py", "--backend", args.backend])
    ]

    results = []
    all_passed = True

    for step_name, cmd in steps:
        status, detail = run_step(step_name, cmd)
        results.append((step_name, status, detail))
        if status != "PASS":
            all_passed = False

    print("\n" + "=" * 80)
    print("                      CONSOLIDATED MASTER VERIFICATION REPORT                   ")
    print("=" * 80)
    print(f"{'Verification Stage':<60} | {'Status':<8}")
    print("-" * 80)

    for step_name, status, detail in results:
        print(f"{step_name:<60} | {status:<8}")

    print("=" * 80)

    if all_passed:
        print("\n" + "*" * 60)
        print("         READY FOR FULL EVALUATION RUN          ")
        print("*" * 60 + "\n")
        sys.exit(0)
    else:
        print("\n[!] VERIFICATION SUITE FAILED: One or more stages failed. See details above.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
