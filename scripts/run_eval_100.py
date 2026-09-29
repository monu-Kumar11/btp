#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Runner for 100-Question PopQA Benchmark (Script: run_eval_100.py)

1. Samples --n_samples (default 100) unique questions from data/popqa/test_popqa.txt to data/popqa/eval_100.txt.
2. Runs all three methods:
   - plain_rag       -> logs/full_plain_rag.csv
   - crag            -> logs/full_crag.csv
   - self_correcting -> logs/full_self_correcting.csv
3. Supports retry and resume logic natively (resumes from last logged query_id if interrupted).
"""

import os
import sys
import argparse
import subprocess

def sample_eval_dataset(source_file="data/popqa/test_popqa.txt", target_file="data/popqa/eval_100.txt", n_samples=100):
    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
        with open(target_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
        q_set = set(l.strip().split(" [SEP]")[0].strip() for l in existing_lines if "[SEP]" in l)
        if len(q_set) == n_samples:
            print(f"Dataset file {target_file} already exists with {len(q_set)} questions.")
            return target_file

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    print(f"Sampling first {n_samples} unique questions from {source_file} into {target_file}...")

    unique_queries = []
    seen_queries = set()

    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip()
            if not c or "[SEP]" not in c:
                continue
            q = c.split(" [SEP]")[0].strip()
            if q not in seen_queries:
                seen_queries.add(q)
                unique_queries.append(q)
                if len(unique_queries) == n_samples:
                    break

    target_queries = set(unique_queries[:n_samples])
    selected_lines = []

    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip()
            if not c or "[SEP]" not in c:
                continue
            q = c.split(" [SEP]")[0].strip()
            if q in target_queries:
                selected_lines.append(line.rstrip("\n"))

    with open(target_file, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_lines) + "\n")

    print(f"Sampled {len(target_queries)} questions ({len(selected_lines)} passage lines) -> {target_file}.")
    return target_file

def parse_args():
    parser = argparse.ArgumentParser(description="Run 100-question evaluation benchmark across plain_rag, crag, self_correcting.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of questions to sample for evaluation")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM generator backend")
    parser.add_argument('--source_file', type=str, default="data/popqa/test_popqa.txt", help="Source PopQA dataset file")
    parser.add_argument('--eval_file', type=str, default="data/popqa/eval_100.txt", help="Target evaluation dataset file")
    return parser.parse_args()

def main():
    args = parse_args()
    python_exe = sys.executable

    eval_dataset = sample_eval_dataset(
        source_file=args.source_file,
        target_file=args.eval_file,
        n_samples=args.n_samples
    )

    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("\n==================================================")
    print(f"STARTING EVALUATION BENCHMARK ({args.n_samples} Questions, Backend: {args.backend})")
    print("==================================================")

    # 1. Plain RAG Run
    print("\n--- Method 1/3: plain_rag ---")
    cmd_plain = [
        python_exe, "scripts/CRAG_Inference.py",
        "--input_file", eval_dataset,
        "--output_file", "output/full_plain_rag_preds.txt",
        "--log_file", "logs/full_plain_rag.csv",
        "--method", "plain_rag",
        "--generator_backend", args.backend,
        "--task", "popqa"
    ]
    subprocess.run(cmd_plain, check=True)

    # 2. CRAG Run
    print("\n--- Method 2/3: crag ---")
    cmd_crag = [
        python_exe, "scripts/CRAG_Inference.py",
        "--input_file", eval_dataset,
        "--output_file", "output/full_crag_preds.txt",
        "--log_file", "logs/full_crag.csv",
        "--method", "crag",
        "--generator_backend", args.backend,
        "--task", "popqa"
    ]
    subprocess.run(cmd_crag, check=True)

    # 3. Self-Correcting Run
    print("\n--- Method 3/3: self_correcting ---")
    cmd_sc = [
        python_exe, "scripts/controller.py",
        "--input_file", eval_dataset,
        "--output_file", "output/full_self_correcting_preds.txt",
        "--log_file", "logs/full_self_correcting.csv",
        "--backend", args.backend,
        "--task", "popqa"
    ]
    subprocess.run(cmd_sc, check=True)

    print("\n==================================================")
    print("ALL THREE EVALUATION RUNS COMPLETED SUCCESSFULLY!")
    print("==================================================")
    print("Output Log Files:")
    print("  - logs/full_plain_rag.csv")
    print("  - logs/full_crag.csv")
    print("  - logs/full_self_correcting.csv")
    print("Prediction Files:")
    print("  - output/full_plain_rag_preds.txt")
    print("  - output/full_crag_preds.txt")
    print("  - output/full_self_correcting_preds.txt")

    # Run check_logs on full logs
    print("\nValidating Log Files...")
    cmd_check = [
        python_exe, "scripts/check_logs.py",
        "--run_log", "logs/full_plain_rag.csv",
        "--controller_log", "logs/full_self_correcting.csv",
        "--expected_questions", str(args.n_samples)
    ]
    subprocess.run(cmd_check, check=False)

    # Run compare_methods on full outputs
    print("\nRunning Method Comparison & Hallucination Rate Benchmark...")
    cmd_compare = [
        python_exe, "scripts/compare_methods.py",
        "--input_file", eval_dataset,
        "--plain_preds", "output/full_plain_rag_preds.txt",
        "--crag_preds", "output/full_crag_preds.txt",
        "--sc_preds", "output/full_self_correcting_preds.txt",
        "--sc_log", "logs/full_self_correcting.csv",
        "--backend", args.backend
    ]
    subprocess.run(cmd_compare, check=False)

if __name__ == "__main__":
    main()
