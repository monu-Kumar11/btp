#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparative Evaluation & Benchmark Script (Script 3)

Loads outputs from plain_rag, crag, and self_correcting, computes groundedness & hallucination rates
using groundedness_eval.py, checks retry triggers, and compares overall metrics across methods.
"""

import os
import sys
import csv
import argparse

# Add scripts dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from groundedness_eval import init_llm, call_eval_llm, parse_verdict

def parse_args():
    parser = argparse.ArgumentParser(description="Compare performance & hallucination rates across RAG methods.")
    parser.add_argument('--input_file', type=str, default="data/popqa/tiny_test.txt", help="Input test file containing query and passages")
    parser.add_argument('--plain_preds', type=str, default="output/plain_rag_preds.txt", help="Plain RAG predictions file")
    parser.add_argument('--crag_preds', type=str, default="output/crag_preds.txt", help="CRAG predictions file")
    parser.add_argument('--sc_preds', type=str, default="output/self_correcting_preds.txt", help="Self-Correcting predictions file")
    parser.add_argument('--sc_log', type=str, default="logs/self_correcting_log.csv", help="Self-Correcting log file")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend for groundedness evaluator")
    return parser.parse_args()

def extract_contexts_from_input(input_file, output_context_file="output/tiny_contexts.txt"):
    os.makedirs(os.path.dirname(output_context_file), exist_ok=True)
    queries = []
    contexts = []
    tmp_psgs = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            c = line.strip()
            if not c:
                continue
            if ' [SEP] ' in c:
                parts = c.split(' [SEP] ')
                q, p = parts[0], parts[1]
            else:
                q, p = c, ""

            if not queries:
                queries.append(q)
                tmp_psgs = [p] if p else []
            else:
                if q != queries[-1]:
                    contexts.append(" ".join(tmp_psgs))
                    queries.append(q)
                    tmp_psgs = [p] if p else []
                else:
                    if p:
                        tmp_psgs.append(p)
        if queries and len(contexts) < len(queries):
            contexts.append(" ".join(tmp_psgs))

    with open(output_context_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(contexts) + "\n")

    return output_context_file, len(queries)

def evaluate_predictions(pred_file, context_file, backend="groq", client_obj=None, model_name=None):
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file {pred_file} not found.")

    with open(pred_file, 'r', encoding='utf-8') as f:
        answers = [l.strip() for l in f.readlines()]
    with open(context_file, 'r', encoding='utf-8') as f:
        contexts = [l.strip() for l in f.readlines()]

    num_eval = min(len(answers), len(contexts))
    if client_obj is None:
        client_obj, model_name = init_llm(backend)

    grounded_count = 0
    ungrounded_count = 0

    for i in range(num_eval):
        ans = answers[i]
        ctx = contexts[i]
        prompt = (
            f"Given this context: {ctx}\n\n"
            f"and this answer: {ans}\n\n"
            "Is the answer fully supported by the context? Respond with GROUNDED or UNGROUNDED and a one-sentence reason."
        )
        resp = call_eval_llm(backend, client_obj, model_name, prompt)
        verdict, reason = parse_verdict(resp)
        if verdict == "GROUNDED":
            grounded_count += 1
        else:
            ungrounded_count += 1

    total = grounded_count + ungrounded_count
    hallucination_rate = (ungrounded_count / total * 100.0) if total > 0 else 0.0
    groundedness_rate = (grounded_count / total * 100.0) if total > 0 else 0.0

    return {
        "total": total,
        "grounded": grounded_count,
        "ungrounded": ungrounded_count,
        "groundedness_rate": groundedness_rate,
        "hallucination_rate": hallucination_rate
    }

def check_self_correcting_retries(sc_log_file):
    if not os.path.exists(sc_log_file):
        return 0
    total_retries = 0
    with open(sc_log_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 10:
                continue
            try:
                iter_num = int(row[9].strip())
                if iter_num > 1:
                    total_retries += 1
            except ValueError:
                pass
    return total_retries

def main():
    args = parse_args()
    print("\n==================================================")
    print("METHOD COMPARISON & HALLUCINATION BENCHMARK (compare_methods.py)")
    print("==================================================")

    ctx_file, num_q = extract_contexts_from_input(args.input_file)
    client_obj, model_name = init_llm(args.backend)

    methods = {
        "plain_rag": args.plain_preds,
        "crag": args.crag_preds,
        "self_correcting": args.sc_preds
    }

    eval_results = {}
    for name, pred_path in methods.items():
        print(f"Evaluating groundedness for {name} ({pred_path})...")
        eval_results[name] = evaluate_predictions(pred_path, ctx_file, backend=args.backend, client_obj=client_obj, model_name=model_name)

    total_sc_retries = check_self_correcting_retries(args.sc_log)

    print("\n----------------------------------------------------------------------------------")
    print(f"{'Method':<18} | {'Total':<6} | {'Grounded':<9} | {'Ungrounded':<10} | {'Grounded Rate':<14} | {'Hallucination Rate':<18}")
    print("----------------------------------------------------------------------------------")
    for name, res in eval_results.items():
        print(f"{name:<18} | {res['total']:<6} | {res['grounded']:<9} | {res['ungrounded']:<10} | {res['groundedness_rate']:<13.1f}% | {res['hallucination_rate']:<17.1f}%")
    print("----------------------------------------------------------------------------------")

    print(f"Self-Correcting Retries Triggered: {total_sc_retries}")

    warnings = []
    plain_hall = eval_results["plain_rag"]["hallucination_rate"]
    sc_hall = eval_results["self_correcting"]["hallucination_rate"]
    if sc_hall > plain_hall:
        warnings.append(f"FLAG WARNING: self_correcting hallucination rate ({sc_hall:.1f}%) is HIGHER than plain_rag ({plain_hall:.1f}%).")

    if total_sc_retries == 0:
        warnings.append("FLAG WARNING: self_correcting triggered ZERO retries across all evaluated questions.")

    if warnings:
        print("\nSUMMARY WARNINGS / FLAGS:")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("\nAll comparative checks passed cleanly! Self-Correcting method outperformed or equaled baselines.")

if __name__ == "__main__":
    main()
