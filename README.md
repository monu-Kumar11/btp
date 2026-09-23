# Corrective Retrieval Augmented Generation (CRAG)

This repository contains the complete, cross-platform working implementation of the paper:
- **[Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884.pdf)** (ICLR / arXiv:2401.15884)
  *Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling*

---

## Overview

Large Language Models (LLMs) inevitably exhibit hallucinations when retrieved documents are irrelevant or inaccurate. **Corrective Retrieval Augmented Generation (CRAG)** is a plug-and-play framework designed to self-correct retrieval results and optimize document utilization for augmented generation:

1. **Lightweight Retrieval Evaluator**: Estimates the overall quality of retrieved documents for an input query and returns a confidence degree.
2. **Action Triggering**: Triggers tailored retrieval actions based on confidence thresholds:
   - **Correct**: Executes Knowledge Refinement via a *Decompose-then-Recompose* algorithm into fine-grained internal knowledge strips.
   - **Incorrect**: Discards static retrieval results and triggers Knowledge Searching via query rewriting and external web/Wikipedia search.
   - **Ambiguous**: Combines refined internal knowledge and external search knowledge.
3. **Augmented Generator**: Synthesizes the final output using the refined/corrected knowledge context.

---

## Key Enhancements & Bug Fixes

This implementation includes critical bug fixes and cross-platform compatibility updates over the original codebase:
- **Windows OS Compatibility**: Fixed POSIX-only Unix signal calls (`signal.SIGALRM`) in page fetching routines to use standard cross-platform HTTP timeouts (`requests.get(url, timeout=10)`).
- **OpenAI API v1+ Support**: Updated keyword extraction routines to support modern `openai>=1.0.0` syntax alongside a built-in local rule-based fallback keyword extractor when API keys are absent.
- **Combined Knowledge Preparation**: Fixed missing imports (`import argparse`), uninitialized variables, and invalid loop syntax (`enumerate(zip(...))`).
- **Inference & Text Formatting**: Stripped leading `#` formatting characters when reading refined knowledge files and added automatic HuggingFace generator fallback when `vLLM` is unavailable.
- **UTF-8 Encoding**: Added explicit `encoding='utf-8'` across all file reading/writing operations.

---

## Quick Start (Running the End-to-End Demo)

To run the complete CRAG pipeline end-to-end (Retrieval Evaluator scoring $\rightarrow$ Action Triggering into `{CORRECT, INCORRECT, AMBIGUOUS}` $\rightarrow$ Knowledge Refinement/Search $\rightarrow$ Answer Generation):

```bash
python demo_crag.py
```

---

## Installation & Requirements

1. **Clone Repository**:
   ```bash
   git clone https://github.com/monu-Kumar11/btp.git
   cd btp
   ```

2. **Python Environment**:
   Python 3.11 is recommended. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Or install core requirements:
   ```bash
   pip install torch transformers jsonlines sentencepiece openai beautifulsoup4 requests pandas scikit-learn tqdm
   ```

---

## Detailed Step-by-Step Usage Guide

### 1. Data Preprocessing
Preprocess dataset questions and retrieval results for PopQA, PubQA, Arc-Challenge, or Bio:
```bash
python scripts/data_process.py --dataset popqa
```

### 2. Retrieval Evaluator Fine-Tuning
To fine-tune the lightweight T5 retrieval evaluator model on query-document pairs:
```bash
python scripts/train_evaluator.py \
  --train_file data/popqa/test_popqa.txt \
  --save_path evaluator_model \
  --batch_size 12 \
  --num_epochs 8 \
  --seed 42
```

### 3. Knowledge Preparation

#### A. Internal Knowledge Refinement (Correct Action)
Decomposes retrieved passages into strips, scores strip relevance with the evaluator, and recomposes top strips:
```bash
python scripts/internal_knowledge_preparation.py \
  --model_path t5-small \
  --input_queries data/popqa/sources \
  --input_retrieval data/popqa/retrieved_psgs \
  --decompose_mode selection \
  --output_file data/popqa/ref/correct
```

#### B. External Knowledge Search (Incorrect Action)
Rewrites queries into search keywords and retrieves external web/Wikipedia content:
```bash
python scripts/external_knowledge_preparation.py \
  --model_path t5-small \
  --input_queries data/popqa/sources \
  --task popqa \
  --mode wiki \
  --output_file data/popqa/ref/incorrect
```

#### C. Combined Knowledge Preparation (Ambiguous Action)
Merges internal refined knowledge and external search knowledge:
```bash
python scripts/combined_knowledge_preparation.py \
  --correct_path data/popqa/ref/correct \
  --incorrect_path data/popqa/ref/incorrect \
  --ambiguous_path data/popqa/ref/ambiguous
```

### 4. CRAG Inference
Run inference to evaluate retrieval confidence, trigger corrective actions, and generate answers using hosted LLM APIs (Groq or Gemini) or local models:

**Using Groq API Backend**:
```bash
export GROQ_API_KEY="your_groq_api_key"
python scripts/CRAG_Inference.py \
  --generator_backend groq \
  --evaluator_path t5-small \
  --input_file data/popqa/test_popqa.txt \
  --output_file data/popqa/output_preds.txt \
  --internal_knowledge_path data/popqa/ref/correct \
  --external_knowledge_path data/popqa/ref/incorrect \
  --combined_knowledge_path data/popqa/ref/ambiguous \
  --task popqa \
  --method crag \
  --ndocs 10 \
  --upper_threshold 0.59 \
  --lower_threshold 0.99
```

**Using Gemini API Backend**:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
python scripts/CRAG_Inference.py \
  --generator_backend gemini \
  --evaluator_path t5-small \
  --input_file data/popqa/test_popqa.txt \
  --output_file data/popqa/output_preds.txt \
  --internal_knowledge_path data/popqa/ref/correct \
  --external_knowledge_path data/popqa/ref/incorrect \
  --combined_knowledge_path data/popqa/ref/ambiguous \
  --task popqa \
  --method crag \
  --ndocs 10 \
  --upper_threshold 0.59 \
  --lower_threshold 0.99
```

**Using Local Backend (vLLM / HuggingFace)**:
```bash
python scripts/CRAG_Inference.py \
  --generator_backend local \
  --generator_path gpt2 \
  --evaluator_path t5-small \
  --input_file data/popqa/test_popqa.txt \
  --output_file data/popqa/output_preds.txt \
  --internal_knowledge_path data/popqa/ref/correct \
  --external_knowledge_path data/popqa/ref/incorrect \
  --combined_knowledge_path data/popqa/ref/ambiguous \
  --task popqa \
  --method crag \
  --ndocs 10 \
  --upper_threshold 0.59 \
  --lower_threshold 0.99
```

### 5. Evaluation Metrics
Evaluate generated responses against ground truth benchmarks:
```bash
python scripts/eval.py \
  --input_file data/popqa/test_popqa.txt \
  --eval_file data/popqa/output_preds.txt \
  --metric match \
  --task popqa
```

---

## Project Structure

```
.
├── demo_crag.py                      # Self-contained end-to-end CRAG pipeline demo
├── README.md                         # Project documentation and usage guide
├── requirements.txt                  # Python dependencies
├── run_crag_inference.sh             # Shell script for CRAG inference
├── run_data_preprocess.sh            # Shell script for data preprocessing
├── run_eval.sh                       # Shell script for evaluation
├── run_evaluator_training.sh         # Shell script for training the evaluator
├── run_knowledge_preparation.sh      # Shell script for knowledge preparation
├── run_selfcrag_preparation.sh       # Shell script for Self-CRAG preparation
├── data/                             # Datasets (PopQA, PubQA, Arc-Challenge, Bio)
│   ├── popqa/
│   ├── pubqa/
│   ├── arc_challenge/
│   └── bio/
└── scripts/                          # Core source code
    ├── CRAG_Inference.py             # Inference loop & generator integration
    ├── combined_knowledge_preparation.py # Ambiguous action knowledge merger
    ├── data_process.py               # Dataset preprocessing & formatting
    ├── eval.py                       # Benchmark evaluation script
    ├── external_knowledge_preparation.py # Web search & external knowledge loader
    ├── internal_knowledge_preparation.py # Decompose-then-recompose refinement
    ├── metrics.py                    # Match & accuracy metric implementations
    ├── train_evaluator.py            # T5 evaluator fine-tuning script
    └── utils.py                      # Helper functions & keyword extraction
```

---

## Citation

If you find this work helpful or use this code, please cite the paper:

```bibtex
@article{yan2024corrective,
  title={Corrective Retrieval Augmented Generation},
  author={Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.15884},
  year={2024}
}
```
