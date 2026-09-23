#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Groundedness and Hallucination Rate Evaluator

Evaluates whether generated answers are fully supported by their retrieved context.
Outputs per-query verdicts and calculates the aggregate hallucination rate (% UNGROUNDED).
"""

import os
import sys
import csv
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate groundedness and hallucination rate of generated RAG/CRAG answers.")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to generated answers file (one answer per line)")
    parser.add_argument('--context_file', type=str, required=True, help="Path to context file (one context per line)")
    parser.add_argument('--output_csv', type=str, default="groundedness_summary.csv", help="Path to output CSV summary file")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend for evaluation")
    parser.add_argument('--model_name', type=str, default=None, help="Model name for LLM evaluator")
    return parser.parse_args()

def init_llm(backend, model_name=None):
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Please install groq package using 'pip install groq'.")
        client = Groq(api_key=api_key)
        model = model_name if model_name else "openai/gpt-oss-120b"
        return client, model
    elif backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai using 'pip install google-generativeai'.")
        genai.configure(api_key=api_key)
        model = model_name if model_name else "gemini-2.5-flash"
        genai_model = genai.GenerativeModel(model)
        return genai_model, model

def call_eval_llm(backend, client_or_model, model_name, prompt):
    if backend == "groq":
        response = client_or_model.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.0,
            max_tokens=150,
        )
        return response.choices[0].message.content or ""
    elif backend == "gemini":
        response = client_or_model.generate_content(prompt)
        return response.text or ""

def parse_verdict(response_text):
    text_upper = response_text.upper()
    if "UNGROUNDED" in text_upper:
        verdict = "UNGROUNDED"
    elif "GROUNDED" in text_upper:
        verdict = "GROUNDED"
    else:
        verdict = "UNGROUNDED"
    
    reason = response_text.strip().replace("\n", " ")
    return verdict, reason

def main():
    args = parse_args()
    
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        answers = [l.strip() for l in f.readlines()]
        
    with open(args.context_file, 'r', encoding='utf-8') as f:
        raw_contexts = [l.strip() for l in f.readlines()]
        contexts = [c[1:] if c.startswith('#') else c for c in raw_contexts]
        
    num_eval = min(len(answers), len(contexts))
    if len(answers) != len(contexts):
        print(f"Warning: Number of answers ({len(answers)}) and contexts ({len(contexts)}) differ. Evaluating first {num_eval} queries.")

    client_obj, model_name = init_llm(args.backend, args.model_name)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    num_grounded = 0
    num_ungrounded = 0

    print(f"\nEvaluating groundedness for {num_eval} queries using {args.backend} ({model_name})...")
    
    for i in tqdm(range(num_eval)):
        ans = answers[i]
        ctx = contexts[i]
        
        if not ans or ans.lower() == "none" or not ctx:
            verdict = "UNGROUNDED"
            reason = "Empty answer or context."
        else:
            prompt = (
                f"Given this context: {ctx}, and this answer: {ans}, "
                "is the answer fully supported by the context? "
                "Respond with GROUNDED or UNGROUNDED and a one-sentence reason."
            )
            try:
                resp_text = call_eval_llm(args.backend, client_obj, model_name, prompt)
                verdict, reason = parse_verdict(resp_text)
            except Exception as e:
                print(f"Error evaluating query {i}: {e}")
                verdict = "UNGROUNDED"
                reason = f"LLM evaluation failed: {e}"

        is_hallucinated = 1 if verdict == "UNGROUNDED" else 0
        if verdict == "GROUNDED":
            num_grounded += 1
        else:
            num_ungrounded += 1

        rows.append({
            "query_id": i,
            "verdict": verdict,
            "is_hallucinated": is_hallucinated,
            "reason": reason,
            "answer": ans,
            "context": ctx[:200] + ("..." if len(ctx) > 200 else "")
        })

    hallucination_rate = (num_ungrounded / num_eval * 100) if num_eval > 0 else 0.0
    grounded_accuracy = (num_grounded / num_eval * 100) if num_eval > 0 else 0.0

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "verdict", "is_hallucinated", "reason", "answer", "context"])
        for r in rows:
            writer.writerow([r["query_id"], r["verdict"], r["is_hallucinated"], r["reason"], r["answer"], r["context"]])
        
        writer.writerow([])
        writer.writerow(["SUMMARY_METRICS", "VALUE"])
        writer.writerow(["total_queries", num_eval])
        writer.writerow(["num_grounded", num_grounded])
        writer.writerow(["num_ungrounded", num_ungrounded])
        writer.writerow(["grounded_accuracy_percent", round(grounded_accuracy, 2)])
        writer.writerow(["hallucination_rate_percent", round(hallucination_rate, 2)])

    print("\n" + "=" * 60)
    print(" GROUNDEDNESS EVALUATION SUMMARY")
    print("=" * 60)
    print(f" Total Queries Evaluated : {num_eval}")
    print(f" Grounded Count          : {num_grounded}")
    print(f" Ungrounded Count        : {num_ungrounded}")
    print(f" Grounded Accuracy (%)   : {grounded_accuracy:.2f}%")
    print(f" Hallucination Rate (%)  : {hallucination_rate:.2f}%")
    print(f" Results Saved To        : {args.output_csv}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
