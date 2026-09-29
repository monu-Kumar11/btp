#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Log Validation Script (Script 2)

Validates CSV log files (logs/run_log.csv and logs/self_correcting_log.csv).
Checks:
- Row counts / unique query_ids match expected test size.
- No null, empty, or zero values in latency_seconds, total_input_tokens, total_output_tokens.
- Flag any invalid or zero metrics.
"""

import os
import sys
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Validate structured CSV run logs.")
    parser.add_argument('--run_log', type=str, default="logs/run_log.csv", help="Path to main run_log.csv")
    parser.add_argument('--controller_log', type=str, default="logs/self_correcting_log.csv", help="Path to self_correcting_log.csv")
    parser.add_argument('--expected_questions', type=int, default=8, help="Expected number of unique questions")
    return parser.parse_args()

def validate_csv_log(file_path, expected_questions, is_controller=False):
    if not os.path.exists(file_path):
        return False, [f"Log file {file_path} does not exist."]

    errors = []
    rows = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False, [f"Log file {file_path} is empty."]
        
        for idx, row in enumerate(reader, start=2):
            if not row or all(c.strip() == "" for c in row):
                continue
            rows.append((idx, row))

    if len(rows) == 0:
        return False, [f"Log file {file_path} contains no data rows."]

    unique_qids = set()
    
    for line_num, row in rows:
        if len(row) < 9:
            errors.append(f"Line {line_num}: Row has only {len(row)} columns, expected at least 9.")
            continue
        
        q_id = row[0].strip()
        method = row[1].strip()
        action = row[2].strip()
        gen_calls = row[3].strip()
        eval_calls = row[4].strip()
        in_tok_str = row[5].strip()
        out_tok_str = row[6].strip()
        latency_str = row[7].strip()
        answer = row[8].strip()

        unique_qids.add(q_id)

        try:
            in_tok = int(in_tok_str)
            if in_tok <= 0:
                errors.append(f"Line {line_num} (q_id={q_id}): total_input_tokens is {in_tok} (must be > 0).")
        except ValueError:
            errors.append(f"Line {line_num} (q_id={q_id}): Invalid total_input_tokens value '{in_tok_str}'.")

        try:
            out_tok = int(out_tok_str)
            if out_tok <= 0:
                errors.append(f"Line {line_num} (q_id={q_id}): total_output_tokens is {out_tok} (must be > 0).")
        except ValueError:
            errors.append(f"Line {line_num} (q_id={q_id}): Invalid total_output_tokens value '{out_tok_str}'.")

        try:
            latency = float(latency_str)
            if latency <= 0.0:
                errors.append(f"Line {line_num} (q_id={q_id}): latency_seconds is {latency} (must be > 0.0).")
        except ValueError:
            errors.append(f"Line {line_num} (q_id={q_id}): Invalid latency_seconds value '{latency_str}'.")

        if not answer:
            errors.append(f"Line {line_num} (q_id={q_id}): final_answer is empty.")

    if len(unique_qids) < expected_questions:
        errors.append(f"Log file {file_path} contains {len(unique_qids)} unique queries, expected {expected_questions}.")

    return len(errors) == 0, errors

def main():
    args = parse_args()
    print("\n==================================================")
    print("LOG VALIDATION CHECK (check_logs.py)")
    print("==================================================")

    all_pass = True

    if os.path.exists(args.run_log):
        ok, errs = validate_csv_log(args.run_log, args.expected_questions)
        status = "PASS" if ok else "FAIL"
        print(f"File: {args.run_log:<30} | Status: {status}")
        if not ok:
            all_pass = False
            for e in errs:
                print(f"  - {e}")
    else:
        print(f"File: {args.run_log:<30} | Status: FAIL (file does not exist)")
        all_pass = False

    if os.path.exists(args.controller_log):
        ok, errs = validate_csv_log(args.controller_log, args.expected_questions, is_controller=True)
        status = "PASS" if ok else "FAIL"
        print(f"File: {args.controller_log:<30} | Status: {status}")
        if not ok:
            all_pass = False
            for e in errs:
                print(f"  - {e}")
    else:
        print(f"File: {args.controller_log:<30} | Status: FAIL (file does not exist)")
        all_pass = False

    print("==================================================")
    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    main()
