#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Claim Critic Unit Test Script (Script 4)

Tests claim_critic.py using 4 hardcoded test cases (2 grounded, 2 ungrounded) to verify
that claim decomposition, entailment checking, and final verdict aggregation function as expected.
"""

import os
import sys
import argparse

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from claim_critic import evaluate_answer_groundedness, init_llm

def parse_args():
    parser = argparse.ArgumentParser(description="Unit test claim critic on 4 canonical test cases.")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend for claim critic")
    return parser.parse_args()

TEST_CASES = [
    {
        "id": 1,
        "name": "Grounded - Capital City",
        "context": "Paris is the capital and most populous city of France.",
        "answer": "Paris is the capital city of France.",
        "expected_verdict": "GROUNDED"
    },
    {
        "id": 2,
        "name": "Grounded - Birthplace",
        "context": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
        "answer": "Einstein was born in Germany in 1879.",
        "expected_verdict": "GROUNDED"
    },
    {
        "id": 3,
        "name": "Ungrounded - Contradiction",
        "context": "The moon orbits Earth at an average distance of 384,400 km every 27.3 days.",
        "answer": "The moon orbits Mars every 10 days.",
        "expected_verdict": "UNGROUNDED"
    },
    {
        "id": 4,
        "name": "Ungrounded - Unsupported Claim",
        "context": "Python was created by Guido van Rossum and first released in 1991.",
        "answer": "Python was created by Guido van Rossum and is the fastest programming language in the world.",
        "expected_verdict": "UNGROUNDED"
    }
]

def main():
    args = parse_args()
    print("\n==================================================")
    print("CLAIM CRITIC UNIT TESTS (test_claim_critic.py)")
    print(f"Backend: {args.backend}")
    print("==================================================")

    client_or_model, model_name = init_llm(args.backend)

    all_passed = True

    for tc in TEST_CASES:
        print(f"\nTest Case {tc['id']}: {tc['name']}")
        print(f"  Context : {tc['context']}")
        print(f"  Answer  : {tc['answer']}")

        res = evaluate_answer_groundedness(
            answer=tc['answer'],
            context=tc['context'],
            backend=args.backend,
            client_or_model=client_or_model,
            model_name=model_name
        )

        actual_verdict = res["final_verdict"]
        passed = (actual_verdict == tc['expected_verdict'])

        print(f"  Claims Extracted  : {res['claims']}")
        print(f"  Claim Verdicts    : {res['claim_verdicts']}")
        print(f"  Failed Claims     : {res['failed_claims']}")
        print(f"  Expected Verdict  : {tc['expected_verdict']}")
        print(f"  Actual Verdict    : {actual_verdict}")
        print(f"  Status            : {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    print("\n==================================================")
    print("CLAIM CRITIC SUMMARY REPORT")
    print("==================================================")
    status_str = "PASS" if all_passed else "FAIL"
    print(f"Overall Claim Critic Unit Tests: {status_str}")
    print("==================================================")

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
