
import argparse
import logging

import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler

from transformers import T5Tokenizer, T5ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM

import csv
import time

# logger = logging.getLogger(__name__)

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

'''
task = ### Instruction: + TASK_INST[task] + ## Input: + question + ### Response:
task = ### Instruction: + TASK_INST[task] + ## Input: + question + choices + ### Response:
'''
def format_prompt(i, task, question, paragraph=None, modelname="selfrag_llama"):
    if paragraph is not None:
        paragraph = ' '.join(paragraph.split(' ')[:])

    instruction = TASK_INST[task] if task in TASK_INST else None
    instruction = instruction + "\n\n## Input:\n\n" + question if instruction is not None else question

    if task == "arc_challenge":
        with open("../data/arc_challenge/choices", 'r', encoding='utf-8') as f:
            choices = f.readlines()[i].strip()
        choices = choices.replace("A: ", "\nA: ")
        choices = choices.replace("B: ", "\nB: ")
        choices = choices.replace("C: ", "\nC: ")
        choices = choices.replace("D: ", "\nD: ")
        choices = choices.replace("E: ", "\nE: ")
        instruction += choices

    if instruction == question:
        # PopQA
        prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Answer the question: " + question
    else:
        if task == "arc_challenge":
            prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\nQuestion: " + question + "\n\nInstruction: Given four answer candidates, A, B, C and D, choose the best answer choice." + "\nChoices:" + choices 

        elif task == "pubqa":
            if modelname == "llama":
                prompt = "Read the documents and answer the question: Is the following statement correct or not? \n\nDocuments: " + paragraph + "\n\nStatement: " + question + "\n\nOnly say true if the statement is true; otherwise say false."
            else:
                prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
                if paragraph is not None:
                    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
            # prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Is the following statement correct or not? Say true if it's correct; otherwise say false. \nStatement: " + question
    # prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    # if paragraph is not None:
    #     prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)

    return prompt

def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

def data_preprocess(file, n_docs):
    # with_label = True
    with_label = False
    queries = []
    passages = []
    tmp_psgs = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for i, line in enumerate(f.readlines()[:]):
                c, l = line.strip().split("\t")
                q, p = c.split(' [SEP] ')
                if queries == []:
                    queries.append(q)
                    tmp_psgs = [p]
                else:
                    if q != queries[-1]:
                        passages.append(' [sep] '.join(tmp_psgs[:n_docs]))
                        queries.append(q)
                        tmp_psgs = [p]
                    else:
                        tmp_psgs.append(p)
                passages.append(' [sep] '.join(tmp_psgs[:n_docs]))
        else:
            for i, line in enumerate(f.readlines()):
                c = line.strip()
                if c.endswith('[SEP]'):
                    c += ' '
                q, p = c.split(' [SEP] ')
                if queries == []:
                    queries.append(q)
                    tmp_psgs = [p]
                else:
                    if q != queries[-1]:
                        passages.append(' [sep] '.join(tmp_psgs[:n_docs]))
                        queries.append(q)
                        tmp_psgs = [p]
                    else:
                        tmp_psgs.append(p)
            passages.append(' [sep] '.join(tmp_psgs[:n_docs]))
    return queries, passages

def get_evaluator_data(file):
    with_label = False
    # with_label = True
    content = []
    label = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for line in f.readlines()[:]:
                c, l = line.split("\t")
                content.append(c)
                label.append((int(l.strip()) - 0.5) * 2)
            return content, label
        else:
            for line in f.readlines():
                content.append(line.strip())
            return content, None

def inference(tokenizer, model, file, device=torch.device("cpu"), n_docs=10):
    model.eval()
    content, label = get_evaluator_data(file)

    preds = []
    scores = []

    for n, c in tqdm(enumerate(content[:])):
        if c.strip().endswith('[SEP]'):
            preds.append(-1)
            scores.append(-1.0)
            continue
        if n % 10 > n_docs-1:
            continue
        test = tokenizer(c, return_tensors="pt",padding="max_length",max_length=512)
        with torch.no_grad():  
            outputs = model(test["input_ids"].to(device), 
                            attention_mask=test["attention_mask"].to(device))
        pred_flat = 1 if outputs["logits"].cpu() > 0 else -1
        scores.append(float(outputs["logits"].cpu()))
        preds.append(pred_flat)
    return scores

def process_flag(scores, n_docs, threshold1, threshold2):
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append('2')
        elif score >= threshold2:
            flags.append('1')
        else:
            flags.append('0')

    tmp_flag = []
    identification_flag = []
    for i, f in enumerate(flags):
        tmp_flag.append(f)
        if i % n_docs == n_docs - 1:
            if '2' in tmp_flag:
                identification_flag.append(2)
            elif '1' in tmp_flag:
                identification_flag.append(1)
            else:
                identification_flag.append(0)
            tmp_flag = []
    return identification_flag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str)
    parser.add_argument('--evaluator_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--internal_knowledge_path', type=str)
    parser.add_argument('--external_knowledge_path', type=str)
    parser.add_argument('--combined_knowledge_path', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--method', type=str, default="default", choices=['rag', 'plain_rag', 'crag', 'self_correcting', 'no_retrieval'])
    parser.add_argument('--max_iterations', type=int, default=3, help="Max retry iterations for self_correcting method")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument('--generator_backend', type=str, default="local", choices=['groq', 'gemini', 'local'],
                        help="Generator backend to use: groq, gemini, or local")
    parser.add_argument("--ndocs", type=int, default=-1,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--upper_threshold", type=float, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--lower_threshold", type=float, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--log_file', type=str, default="../logs/run_log.csv",
                        help="Path to output CSV log file for query execution metrics")
    args = parser.parse_args()
    args.lower_threshold = -args.lower_threshold

    if args.method == 'self_correcting':
        from controller import run_self_correcting_loop
        run_self_correcting_loop(
            input_file=args.input_file,
            context_file=args.internal_knowledge_path,
            output_file=args.output_file,
            log_file=args.log_file,
            backend=args.generator_backend,
            model_name=args.generator_path,
            max_iterations=args.max_iterations,
            task=args.task
        )
        return

    # Ensure log directory exists and header is written if file is new
    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0
    if not file_exists:
        with open(args.log_file, 'a', newline='', encoding='utf-8') as log_f:
            writer = csv.writer(log_f)
            writer.writerow([
                "query_id",
                "method",
                "action_triggered",
                "num_generation_calls",
                "num_evaluator_calls",
                "total_input_tokens",
                "total_output_tokens",
                "latency_seconds",
                "final_answer"
            ])

    # Initialize Generator backend (Groq API, Gemini API, or local vLLM/HuggingFace)
    if args.generator_backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Please install the groq package using 'pip install groq'.")
        groq_client = Groq(api_key=api_key)
        model_name = args.generator_path if (args.generator_path and "/" not in args.generator_path) else "openai/gpt-oss-120b"
    elif args.generator_backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai using 'pip install google-generativeai'.")
        genai.configure(api_key=api_key)
        model_name = args.generator_path if (args.generator_path and "/" not in args.generator_path) else "gemini-2.5-flash"
        gemini_model = genai.GenerativeModel(model_name)
    else:
        try:
            from vllm import LLM, SamplingParams
            generator = LLM(model=args.generator_path, dtype="half")
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
            is_vllm = True
        except (ImportError, Exception) as err:
            print(f"vLLM not available or failed to load ({err}), falling back to HuggingFace pipeline.")
            from transformers import pipeline
            device_id = 0 if torch.cuda.is_available() else -1
            generator = pipeline("text-generation", model=args.generator_path, device=device_id)
            is_vllm = False

    queries, passages = data_preprocess(args.input_file, args.ndocs)

    if args.method in ['rag', 'plain_rag']:
        paragraphs = passages
    elif args.method == 'crag':
        tokenizer = T5Tokenizer.from_pretrained(args.evaluator_path)
        model = T5ForSequenceClassification.from_pretrained(args.evaluator_path, num_labels=1)
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        scores = inference(
            tokenizer=tokenizer, 
            model=model, 
            file=args.input_file,
            device=device, 
            n_docs=args.ndocs
        )
        identification_flag = process_flag(scores, args.ndocs, args.upper_threshold, args.lower_threshold)

        with open(args.internal_knowledge_path, 'r', encoding='utf-8') as in_f, open(args.external_knowledge_path, 'r', encoding='utf-8') as ex_f, open(args.combined_knowledge_path, 'r', encoding='utf-8') as comb_f:
            internal_paragraphs = [l.strip()[1:] if l.strip().startswith('#') else l.strip() for l in in_f.readlines()]
            external_paragraphs = [l.strip()[1:] if l.strip().startswith('#') else l.strip() for l in ex_f.readlines()]
            combined_paragraphs = [l.strip()[1:] if l.strip().startswith('#') else l.strip() for l in comb_f.readlines()]

        paragraphs = []
        n = 0
        for flag, i, e, c in zip(identification_flag, internal_paragraphs, external_paragraphs, combined_paragraphs):
            if flag == 0:
                paragraphs.append(e) # incorrect
            elif flag == 1:
                paragraphs.append(c) # ambiguous
            elif flag == 2:
                paragraphs.append(i) # correct
            n += 1
    
    def generate_response(prompt):
        raw_text = ""
        in_tok = 0
        out_tok = 0

        if args.generator_backend == "groq":
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.0,
                max_tokens=100,
            )
            raw_text = chat_completion.choices[0].message.content or ""
            if hasattr(chat_completion, 'usage') and chat_completion.usage:
                in_tok = chat_completion.usage.prompt_tokens or len(prompt.split())
                out_tok = chat_completion.usage.completion_tokens or len(raw_text.split())
            else:
                in_tok = len(prompt.split())
                out_tok = len(raw_text.split())

        elif args.generator_backend == "gemini":
            response = gemini_model.generate_content(prompt)
            raw_text = response.text or ""
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                in_tok = getattr(response.usage_metadata, 'prompt_token_count', len(prompt.split()))
                out_tok = getattr(response.usage_metadata, 'candidates_token_count', len(raw_text.split()))
            else:
                in_tok = len(prompt.split())
                out_tok = len(raw_text.split())

        else:
            if is_vllm:
                pred = generator.generate([prompt], sampling_params)
                raw_text = pred[0].outputs[0].text or ""
                in_tok = len(prompt.split())
                out_tok = len(raw_text.split())
            else:
                out = generator(prompt, max_new_tokens=100, do_sample=False)
                generated_text = out[0]['generated_text']
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                raw_text = generated_text or ""
                in_tok = len(prompt.split())
                out_tok = len(raw_text.split())

        return raw_text, in_tok, out_tok

    preds = []
    action_map = {0: "incorrect", 1: "ambiguous", 2: "correct"}
    modelname = "selfrag_llama" if (args.generator_path and "selfrag" in args.generator_path) else "llama"

    if args.method != 'no_retrieval':
        for i, (q, p) in tqdm(enumerate(zip(queries, paragraphs))):
            t_start = time.time()
            prompt = format_prompt(i, args.task, q, p, modelname)
            raw_text, in_tok, out_tok = generate_response(prompt)
            final_ans = postprocess_answer_option_conditioned(raw_text)
            preds.append(final_ans)
            latency = round(time.time() - t_start, 4)

            if args.method == 'crag':
                flag_val = identification_flag[i] if i < len(identification_flag) else 0
                action_str = action_map.get(flag_val, "none")
                eval_calls = args.ndocs if args.ndocs > 0 else 10
            else:
                action_str = "none"
                eval_calls = 0

            with open(args.log_file, 'a', newline='', encoding='utf-8') as log_f:
                writer = csv.writer(log_f)
                writer.writerow([
                    i,
                    args.method,
                    action_str,
                    1,
                    eval_calls,
                    in_tok,
                    out_tok,
                    latency,
                    final_ans
                ])
                log_f.flush()
    else:
        for i, q in tqdm(enumerate(queries)):
            t_start = time.time()
            p = None
            prompt = format_prompt(i, args.task, q, p, modelname)
            raw_text, in_tok, out_tok = generate_response(prompt)
            final_ans = postprocess_answer_option_conditioned(raw_text)
            preds.append(final_ans)
            latency = round(time.time() - t_start, 4)

            with open(args.log_file, 'a', newline='', encoding='utf-8') as log_f:
                writer = csv.writer(log_f)
                writer.writerow([
                    i,
                    args.method,
                    "none",
                    1,
                    0,
                    in_tok,
                    out_tok,
                    latency,
                    final_ans
                ])
                log_f.flush()

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))


if __name__ == '__main__':
    main()