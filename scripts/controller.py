#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Self-Correcting Iterative RAG Controller

Orchestrates an iterative self-correcting RAG loop:
1. Retrieve / load initial context
2. Generate candidate answer via hosted LLM (Groq / Gemini) or local model
3. Run claim_critic fine-grained entailment check (from scripts/claim_critic.py)
4. If UNGROUNDED and iteration < max_iterations:
   - Reformulate search query via LLM call based on failed claims
   - Re-retrieve external web/Wikipedia knowledge
   - Retry answer generation (loop back to step 2)
5. If still UNGROUNDED after max_iterations, return answer flagged as [Low Confidence].

Logs every iteration to CSV with schema:
query_id, method, action_triggered, num_generation_calls, num_evaluator_calls,
total_input_tokens, total_output_tokens, latency_seconds, final_answer, retry_iteration
"""

import os
import sys
import csv
import time
import argparse
from tqdm import tqdm

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import extract_keywords
from external_knowledge_preparation import test_page_loader
from claim_critic import evaluate_answer_groundedness, init_llm, _call_llm

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Correcting RAG Orchestration Controller")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input test queries file")
    parser.add_argument('--context_file', type=str, default=None, help="Path to initial context file (optional)")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output predictions file")
    parser.add_argument('--log_file', type=str, default="../logs/self_correcting_log.csv", help="Path to output CSV log file")
    parser.add_argument('--backend', type=str, default="groq", choices=['groq', 'gemini'], help="LLM backend for generation & critique")
    parser.add_argument('--model_name', type=str, default=None, help="Model name for LLM generator")
    parser.add_argument('--max_iterations', type=int, default=3, help="Maximum number of retry iterations per query")
    parser.add_argument('--task', type=str, default="popqa", help="Task name")
    return parser.parse_args()

def load_queries_and_contexts(input_file, context_file=None):
    queries = []
    contexts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            c = line.strip()
            if not c:
                continue
            if ' [SEP] ' in c:
                parts = c.split(' [SEP] ')
                queries.append(parts[0])
                contexts.append(parts[1])
            else:
                queries.append(c)
                contexts.append("")

    if context_file and os.path.exists(context_file):
        with open(context_file, 'r', encoding='utf-8') as f:
            ext_contexts = [l.strip()[1:] if l.strip().startswith('#') else l.strip() for l in f.readlines()]
            if len(ext_contexts) == len(queries):
                contexts = ext_contexts

    return queries, contexts

def format_generation_prompt(query, context, task="popqa"):
    if context:
        return f"Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {context}\n\nInstruction: Answer the question: {query}"
    else:
        return f"Answer the question: {query}"

def generate_llm_response(prompt, backend, client_or_model, model_name):
    t0 = time.time()
    raw_text = ""
    in_tok = 0
    out_tok = 0

    if backend == "groq":
        chat_completion = client_or_model.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.0,
            max_tokens=150,
        )
        raw_text = chat_completion.choices[0].message.content or ""
        if hasattr(chat_completion, 'usage') and chat_completion.usage:
            in_tok = chat_completion.usage.prompt_tokens or len(prompt.split())
            out_tok = chat_completion.usage.completion_tokens or len(raw_text.split())
        else:
            in_tok = len(prompt.split())
            out_tok = len(raw_text.split())

    elif backend == "gemini":
        response = client_or_model.generate_content(prompt)
        raw_text = response.text or ""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            in_tok = getattr(response.usage_metadata, 'prompt_token_count', len(prompt.split()))
            out_tok = getattr(response.usage_metadata, 'candidates_token_count', len(raw_text.split()))
        else:
            in_tok = len(prompt.split())
            out_tok = len(raw_text.split())

    return raw_text.strip(), in_tok, out_tok

def search_external_web(search_query):
    wiki_url = f"https://en.wikipedia.org/wiki/{search_query.replace(' ', '_')}"
    paras = test_page_loader(wiki_url)
    if paras:
        return "; ".join(paras[:3])
    else:
        return f"Retrieved summary for {search_query}"

def log_csv_row(log_file, row_data):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.exists(log_file) and os.path.getsize(log_file) > 0
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "query_id",
                "method",
                "action_triggered",
                "num_generation_calls",
                "num_evaluator_calls",
                "total_input_tokens",
                "total_output_tokens",
                "latency_seconds",
                "final_answer",
                "retry_iteration"
            ])
        writer.writerow(row_data)
        f.flush()

def run_self_correcting_loop(input_file, context_file, output_file, log_file, backend="groq", model_name=None, max_iterations=3, task="popqa"):
    queries, initial_contexts = load_queries_and_contexts(input_file, context_file)
    client_or_model, model_name = init_llm(backend, model_name)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    final_predictions = []

    print(f"\nStarting Self-Correcting RAG Controller for {len(queries)} queries (Max Iterations: {max_iterations}, Backend: {backend})...\n")

    for q_id, (query, init_ctx) in enumerate(tqdm(zip(queries, initial_contexts), total=len(queries))):
        current_context = init_ctx
        iter_count = 0
        final_answer = ""
        is_grounded = False

        while iter_count < max_iterations:
            iter_count += 1
            t_start = time.time()

            # Step 1 & 2: Generate answer
            gen_prompt = format_generation_prompt(query, current_context, task)
            candidate_ans, in_tok, out_tok = generate_llm_response(gen_prompt, backend, client_or_model, model_name)

            # Step 3: Run claim_critic entailment check
            critic_res = evaluate_answer_groundedness(candidate_ans, current_context, backend=backend, client_or_model=client_or_model, model_name=model_name)
            is_grounded = critic_res["is_grounded"]
            failed_claims = critic_res["failed_claims"]
            num_claims = len(critic_res["claims"])

            action_str = "correct" if is_grounded else ("incorrect" if iter_count == 1 else "ambiguous")
            latency = round(time.time() - t_start, 4)

            # Log this iteration row to CSV
            log_csv_row(log_file, [
                q_id,
                "self_correcting",
                action_str,
                1,                # num_generation_calls
                num_claims,       # num_evaluator_calls (number of claim entailment checks)
                in_tok,
                out_tok,
                latency,
                candidate_ans,
                iter_count
            ])

            if is_grounded:
                final_answer = candidate_ans
                break
            else:
                # Step 4: If UNGROUNDED and retries remaining, reformulate query and re-retrieve
                if iter_count < max_iterations:
                    failed_str = "; ".join(failed_claims)
                    rewrite_prompt = (
                        f"Rewrite this search query to better find information supporting: {failed_str}.\n"
                        f"Original query: {query}\n"
                        "Return only the rewritten search query."
                    )
                    rewritten_query, _, _ = generate_llm_response(rewrite_prompt, backend, client_or_model, model_name)
                    new_kw = rewritten_query.strip().split("\n")[0].replace('"', '').strip()

                    # Re-retrieve web/wikipedia knowledge
                    new_knowledge = search_external_web(new_kw)
                    current_context = f"{current_context} [sep] Retried Knowledge ({new_kw}): {new_knowledge}"
                else:
                    # Step 5: Still UNGROUNDED after max_iterations -> return low-confidence flag
                    final_answer = f"[Low Confidence] {candidate_ans}"

        final_predictions.append(final_answer)

    # Save final output predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_predictions))

    print(f"\nSelf-Correcting RAG complete! Output written to {output_file} and log to {log_file}.\n")

def main():
    args = parse_args()
    run_self_correcting_loop(
        input_file=args.input_file,
        context_file=args.context_file,
        output_file=args.output_file,
        log_file=args.log_file,
        backend=args.backend,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        task=args.task
    )

if __name__ == '__main__':
    main()
