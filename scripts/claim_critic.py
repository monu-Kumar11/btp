#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Claim Critic & Fine-Grained Groundedness Evaluator

Decomposes generated answers into atomic factual claims, checks claim-level entailment
(ENTAILMENT, CONTRADICTION, NEUTRAL) against retrieved context, and aggregates into a final
verdict (GROUNDED vs UNGROUNDED).

This module is designed to be both a standalone CLI evaluator and an importable critic module
for retry controllers.
"""

import os
import sys
import csv
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Claim-level fine-grained groundedness and entailment critic.")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to predictions file (one answer per line)")
    parser.add_argument('--context_file', type=str, required=True, help="Path to context file (one context per line)")
    parser.add_argument('--output_csv', type=str, default="../logs/claim_critic_results.csv", help="Path to output CSV file")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend")
    parser.add_argument('--model_name', type=str, default=None, help="LLM model name")
    return parser.parse_args()

def init_llm(backend="groq", model_name=None):
    """
    Initialize LLM client for Groq or Gemini.
    """
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

def _call_llm(backend, client_or_model, model_name, prompt):
    """
    Internal helper to execute LLM prompt.
    """
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

def decompose_answer_into_claims(answer, backend="groq", client_or_model=None, model_name=None):
    """
    Deconstructs a generated answer into 2-5 distinct factual claims.
    """
    if not answer or len(answer.strip()) == 0:
        return []

    if client_or_model is None:
        client_or_model, model_name = init_llm(backend, model_name)

    prompt = (
        f"Decompose the following text into 2-5 distinct, standalone factual claims.\n"
        f"Output each claim on a new line starting with a dash (-).\n\n"
        f"Text: {answer}\n"
    )

    try:
        raw_output = _call_llm(backend, client_or_model, model_name, prompt)
        claims = []
        for line in raw_output.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                claim_text = line.lstrip("-* ").strip()
                if claim_text:
                    claims.append(claim_text)
        if not claims:
            claims = [answer.strip()]
        return claims
    except Exception as e:
        print(f"Warning: Claim decomposition failed ({e}). Treating full answer as single claim.")
        return [answer.strip()]

def check_claim_entailment(claim, context, backend="groq", client_or_model=None, model_name=None):
    """
    Checks if context ENTAILS, CONTRADICTS, or is NEUTRAL toward a specific claim.
    Returns: 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'.
    """
    if not claim or not context:
        return "NEUTRAL"

    if client_or_model is None:
        client_or_model, model_name = init_llm(backend, model_name)

    prompt = (
        f"Given this context: {context}\n\n"
        f"and this factual claim: {claim}\n\n"
        "Does the context ENTAIL, CONTRADICT, or is NEUTRAL toward the claim?\n"
        "Respond with exactly one word: ENTAILMENT, CONTRADICTION, or NEUTRAL."
    )

    try:
        response_text = _call_llm(backend, client_or_model, model_name, prompt).upper().strip()
        if "CONTRADICT" in response_text:
            return "CONTRADICTION"
        elif "ENTAIL" in response_text:
            return "ENTAILMENT"
        else:
            return "NEUTRAL"
    except Exception as e:
        print(f"Warning: Entailment check failed for claim '{claim}' ({e}). Defaulting to NEUTRAL.")
        return "NEUTRAL"

def evaluate_answer_groundedness(answer, context, backend="groq", client_or_model=None, model_name=None):
    """
    Main reusable critic function for controllers/retry logic.
    Returns:
        dict: {
            "claims": list[str],
            "claim_verdicts": list[str],
            "final_verdict": "GROUNDED" | "UNGROUNDED",
            "failed_claims": list[str],
            "is_grounded": bool
        }
    """
    if not answer or not context or len(answer.strip()) == 0 or len(context.strip()) == 0:
        return {
            "claims": [],
            "claim_verdicts": [],
            "final_verdict": "UNGROUNDED",
            "failed_claims": ["Empty answer or context"],
            "is_grounded": False
        }

    if client_or_model is None:
        client_or_model, model_name = init_llm(backend, model_name)

    claims = decompose_answer_into_claims(answer, backend=backend, client_or_model=client_or_model, model_name=model_name)
    verdicts = []
    failed_claims = []

    for claim in claims:
        verdict = check_claim_entailment(claim, context, backend=backend, client_or_model=client_or_model, model_name=model_name)
        verdicts.append(verdict)
        if verdict != "ENTAILMENT":
            failed_claims.append(f"{claim} [{verdict}]")

    is_grounded = (len(failed_claims) == 0) and (len(claims) > 0)
    final_verdict = "GROUNDED" if is_grounded else "UNGROUNDED"

    return {
        "claims": claims,
        "claim_verdicts": verdicts,
        "final_verdict": final_verdict,
        "failed_claims": failed_claims,
        "is_grounded": is_grounded
    }

def main():
    args = parse_args()

    with open(args.pred_file, 'r', encoding='utf-8') as f:
        answers = [l.strip() for l in f.readlines()]

    with open(args.context_file, 'r', encoding='utf-8') as f:
        raw_contexts = [l.strip() for l in f.readlines()]
        contexts = [c[1:] if c.startswith('#') else c for c in raw_contexts]

    num_eval = min(len(answers), len(contexts))
    if len(answers) != len(contexts):
        print(f"Warning: Number of predictions ({len(answers)}) and contexts ({len(contexts)}) differ. Evaluating first {num_eval} queries.")

    client_or_model, model_name = init_llm(args.backend, args.model_name)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    grounded_count = 0
    ungrounded_count = 0

    print(f"\nRunning Claim-Level Groundedness Critic on {num_eval} queries using {args.backend} ({model_name})...\n")

    for i in tqdm(range(num_eval)):
        ans = answers[i]
        ctx = contexts[i]

        res = evaluate_answer_groundedness(ans, ctx, backend=args.backend, client_or_model=client_or_model, model_name=model_name)

        if res["is_grounded"]:
            grounded_count += 1
        else:
            ungrounded_count += 1

        rows.append({
            "query_id": i,
            "claims_list": " | ".join(res["claims"]),
            "per_claim_verdicts": " | ".join(res["claim_verdicts"]),
            "final_verdict": res["final_verdict"],
            "failed_claims": " | ".join(res["failed_claims"])
        })

    # Write per-query CSV results
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "claims_list", "per_claim_verdicts", "final_verdict", "failed_claims"])
        for r in rows:
            writer.writerow([r["query_id"], r["claims_list"], r["per_claim_verdicts"], r["final_verdict"], r["failed_claims"]])

        # Append summary metrics block
        writer.writerow([])
        writer.writerow(["SUMMARY_METRICS", "VALUE"])
        writer.writerow(["total_queries", num_eval])
        writer.writerow(["num_grounded", grounded_count])
        writer.writerow(["num_ungrounded", ungrounded_count])
        writer.writerow(["grounded_accuracy_percent", round(grounded_count / num_eval * 100, 2) if num_eval > 0 else 0.0])
        writer.writerow(["hallucination_rate_percent", round(ungrounded_count / num_eval * 100, 2) if num_eval > 0 else 0.0])

    acc = (grounded_count / num_eval * 100) if num_eval > 0 else 0.0
    halluc = (ungrounded_count / num_eval * 100) if num_eval > 0 else 0.0

    print("\n" + "=" * 65)
    print(" CLAIM CRITIC GROUNDEDNESS EVALUATION SUMMARY")
    print("=" * 65)
    print(f" Total Queries Evaluated : {num_eval}")
    print(f" Grounded Answers        : {grounded_count}")
    print(f" Ungrounded Answers      : {ungrounded_count}")
    print(f" Grounded Accuracy (%)   : {acc:.2f}%")
    print(f" Hallucination Rate (%)  : {halluc:.2f}%")
    print(f" CSV Results Saved To    : {args.output_csv}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
