"""
Corrective Retrieval Augmented Generation (CRAG) End-to-End Demonstration Script

This script demonstrates the full pipeline of CRAG:
1. Query & Retrieval Document Preparation
2. Lightweight Retrieval Evaluator Scoring
3. Discriminative Action Triggering ({CORRECT, INCORRECT, AMBIGUOUS})
4. Knowledge Refinement (Decompose-then-Recompose for Correct action)
5. Knowledge Searching (Query Rewriting & External Search for Incorrect action)
6. Combined Knowledge Integration (for Ambiguous action)
7. Augmented Generation & Evaluation
"""

import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForSequenceClassification

# Add scripts directory to module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from utils import select_relevants, extract_keywords
from internal_knowledge_preparation import extract_strips_from_psg
from external_knowledge_preparation import test_page_loader

def evaluate_retrieval(query, passages, model_name="t5-small", device="cpu"):
    """
    Step 2: Retrieval Evaluator
    Scores relevance of each query-passage pair using T5 evaluator model.
    Returns relevance scores.
    """
    print("\n[Step 2] Running Retrieval Evaluator...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)
    model.eval()

    scores = []
    for idx, psg in enumerate(passages):
        input_content = query + " [SEP] " + psg
        inputs = tokenizer(input_content, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            score = float(outputs.logits.cpu().squeeze())
        scores.append(score)
        print(f"  Passage {idx+1} Relevance Score: {score:.4f}")
    return scores, model, tokenizer

def action_trigger(scores, upper_threshold=0.5, lower_threshold=-0.5):
    """
    Step 3: Action Trigger
    Determines action: CORRECT (2), AMBIGUOUS (1), or INCORRECT (0)
    """
    print("\n[Step 3] Evaluator Action Triggering...")
    has_correct = any(s >= upper_threshold for s in scores)
    all_incorrect = all(s < lower_threshold for s in scores)

    if has_correct:
        action = "CORRECT"
        action_code = 2
    elif all_incorrect:
        action = "INCORRECT"
        action_code = 0
    else:
        action = "AMBIGUOUS"
        action_code = 1

    print(f"  Triggered Action: [{action}] (Upper Threshold: {upper_threshold}, Lower Threshold: {lower_threshold})")
    return action, action_code

def refine_internal_knowledge(query, passages, model, tokenizer, device="cpu", mode="excerption"):
    """
    Step 4: Knowledge Refinement (Decompose-then-Recompose)
    """
    print("\n[Step 4] Running Knowledge Refinement (Decompose-then-Recompose)...")
    strips = []
    for psg in passages:
        strips += extract_strips_from_psg(psg, mode=mode)
    
    print(f"  Decomposed into {len(strips)} knowledge strips.")
    refined_text, idxs = select_relevants(strips, query, tokenizer, model, device, top_n=3)
    print(f"  Selected Top Knowledge Strips: {refined_text[:150]}...")
    return refined_text

def search_external_knowledge(query, task="popqa"):
    """
    Step 5: Knowledge Search (External Web Search / Wikipedia Integration)
    """
    print("\n[Step 5] Running External Knowledge Search...")
    keywords = extract_keywords([query], task, openai_key="")
    search_query = keywords[0]
    print(f"  Extracted Keywords for Search: '{search_query}'")
    
    # Example Wikipedia page fetch using test_page_loader
    wiki_url = f"https://en.wikipedia.org/wiki/{search_query.replace(' ', '_')}"
    print(f"  Fetching web page: {wiki_url}")
    paragraphs = test_page_loader(wiki_url)
    
    if paragraphs:
        external_text = "; ".join(paragraphs[:3])
        print(f"  Fetched {len(paragraphs)} external paragraphs.")
    else:
        external_text = f"External Knowledge Summary for {search_query}"
        print("  Fallback: External knowledge summary created.")
    return external_text

def generate_answer(query, context, task="popqa"):
    """
    Step 6: Final Generation with Augmented Knowledge
    """
    print("\n[Step 6] Final Answer Generation...")
    prompt = f"Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {context}\n\nInstruction: Answer the question: {query}"
    print(f"--- Prompt Sent to Generator ---\n{prompt[:300]}...\n--------------------------------")
    
    answer = f"Generated answer for query '{query}' based on retrieved/refined knowledge context."
    return answer

def run_crag_demo():
    print("=" * 70)
    print(" CORRECTIVE RETRIEVAL AUGMENTED GENERATION (CRAG) DEMONSTRATION")
    print("=" * 70)

    # Sample input query and retrieved documents (from PopQA benchmark dataset format)
    sample_query = "What is Henry Feilden's occupation?"
    sample_passages = [
        "Henry Master Feilden (1818 - 1875) was a British Conservative Party politician.",
        "Henry Feilden was educated at Eton and Christ Church, Oxford. He served as MP for Blackburn.",
        "Batman (1989 film): of the murder of Bruce Wayne's parents. When Hamm's script was rewritten...",
    ]

    print(f"\n[Step 1] Input Query: '{sample_query}'")
    print(f"  Retrieved Documents Count: {len(sample_passages)}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Compute Device: {device}")

    # Step 2: Evaluation
    scores, model, tokenizer = evaluate_retrieval(sample_query, sample_passages, model_name="t5-small", device=device)

    # Step 3: Action Triggering
    action, action_code = action_trigger(scores, upper_threshold=0.0, lower_threshold=-1.0)

    # Steps 4 & 5: Corrective Knowledge Preparation based on Action
    if action == "CORRECT":
        context = refine_internal_knowledge(sample_query, sample_passages, model, tokenizer, device=device)
    elif action == "INCORRECT":
        context = search_external_knowledge(sample_query)
    else: # AMBIGUOUS
        internal_k = refine_internal_knowledge(sample_query, sample_passages, model, tokenizer, device=device)
        external_k = search_external_knowledge(sample_query)
        context = f"Internal Knowledge: {internal_k} [sep] External Knowledge: {external_k}"

    # Step 6: Final Generation
    final_answer = generate_answer(sample_query, context)
    print(f"\n[Final Output]: {final_answer}")
    print("=" * 70)

if __name__ == '__main__':
    run_crag_demo()
