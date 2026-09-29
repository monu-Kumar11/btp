#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final Check Verification Script (Script 1)

Runs all three methods (plain_rag, crag, self_correcting) on a small test file (8 questions).
Creates data/popqa/tiny_test.txt if it doesn't exist.
Reports PASS/FAIL per method based on execution success and non-empty answer outputs.
"""

import os
import sys
import argparse
import subprocess

def create_tiny_test_if_missing(tiny_path="data/popqa/tiny_test.txt", full_path="data/popqa/test_popqa.txt", num_questions=8):
    if os.path.exists(tiny_path) and os.path.getsize(tiny_path) > 0:
        return

    os.makedirs(os.path.dirname(tiny_path), exist_ok=True)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Source file {full_path} not found to create tiny_test.txt")

    print(f"Creating {tiny_path} with first {num_questions} questions from {full_path}...")
    selected_lines = []
    seen_queries = set()

    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip()
            if not c:
                continue
            q = c.split(" [SEP] ")[0] if " [SEP] " in c else c
            if q not in seen_queries:
                if len(seen_queries) >= num_questions:
                    break
                seen_queries.add(q)
            selected_lines.append(line.rstrip("\n"))

    with open(tiny_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_lines) + "\n")
    print(f"Created {tiny_path} with {len(seen_queries)} questions ({len(selected_lines)} total lines).")

def parse_args():
    parser = argparse.ArgumentParser(description="Run final check verification across plain_rag, crag, self_correcting methods.")
    parser.add_argument('--input_file', type=str, default="data/popqa/tiny_test.txt", help="Input test file")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM generator backend")
    parser.add_argument('--num_questions', type=int, default=8, help="Number of questions for tiny test dataset")
    return parser.parse_args()

def verify_output_file(output_file, expected_count=8):
    if not os.path.exists(output_file):
        return False, f"Output file {output_file} was not created."
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    if len(lines) == 0:
        return False, f"Output file {output_file} is empty."
    if len(lines) != expected_count:
        return False, f"Output file {output_file} has {len(lines)} lines, expected {expected_count}."
    empty_lines = [i for i, line in enumerate(lines) if len(line) == 0]
    if empty_lines:
        return False, f"Output file {output_file} contains empty predictions at indices: {empty_lines}."
    return True, "OK"

def main():
    args = parse_args()
    create_tiny_test_if_missing(args.input_file, num_questions=args.num_questions)

    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    results = {}
    python_exe = sys.executable

    # Run Method 1: plain_rag
    print("\n==================================================")
    print("Testing Method 1: plain_rag")
    print("==================================================")
    cmd_plain = [
        python_exe, "scripts/CRAG_Inference.py",
        "--input_file", args.input_file,
        "--output_file", "output/plain_rag_preds.txt",
        "--log_file", "logs/run_log.csv",
        "--method", "plain_rag",
        "--generator_backend", args.backend,
        "--task", "popqa"
    ]
    try:
        ret = subprocess.run(cmd_plain, check=True)
        ok, msg = verify_output_file("output/plain_rag_preds.txt", expected_count=args.num_questions)
        results["plain_rag"] = ("PASS" if ok else "FAIL", msg)
    except Exception as e:
        results["plain_rag"] = ("FAIL", str(e))

    # Run Method 2: crag
    print("\n==================================================")
    print("Testing Method 2: crag")
    print("==================================================")
    cmd_crag = [
        python_exe, "scripts/CRAG_Inference.py",
        "--input_file", args.input_file,
        "--output_file", "output/crag_preds.txt",
        "--log_file", "logs/run_log.csv",
        "--method", "crag",
        "--generator_backend", args.backend,
        "--task", "popqa"
    ]
    try:
        ret = subprocess.run(cmd_crag, check=True)
        ok, msg = verify_output_file("output/crag_preds.txt", expected_count=args.num_questions)
        results["crag"] = ("PASS" if ok else "FAIL", msg)
    except Exception as e:
        results["crag"] = ("FAIL", str(e))

    # Run Method 3: self_correcting
    print("\n==================================================")
    print("Testing Method 3: self_correcting")
    print("==================================================")
    cmd_sc = [
        python_exe, "scripts/controller.py",
        "--input_file", args.input_file,
        "--output_file", "output/self_correcting_preds.txt",
        "--log_file", "logs/self_correcting_log.csv",
        "--backend", args.backend,
        "--task", "popqa"
    ]
    try:
        ret = subprocess.run(cmd_sc, check=True)
        ok, msg = verify_output_file("output/self_correcting_preds.txt", expected_count=args.num_questions)
        results["self_correcting"] = ("PASS" if ok else "FAIL", msg)
    except Exception as e:
        results["self_correcting"] = ("FAIL", str(e))

    print("\n==================================================")
    print("FINAL CHECK SUMMARY REPORT")
    print("==================================================")
    all_pass = True
    for method, (status, detail) in results.items():
        print(f"Method: {method:<16} | Status: {status:<4} | Details: {detail}")
        if status != "PASS":
            all_pass = False

    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
