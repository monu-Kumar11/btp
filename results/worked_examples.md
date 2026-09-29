# RAG Evaluation: Worked Examples & 'Caught and Fixed' Cases

This document highlights queries where the baseline **`plain_rag`** produced an ungrounded or hallucinated answer, but the **`self_correcting`** framework successfully caught the error via the claim critic, triggered iterative retrieval/reformulation, and produced a verified **GROUNDED** final answer.

---

## Case Study 1: Query #2

**Question**: `What is Edward Corser's occupation?`

- **Plain RAG Answer** (UNGROUNDED):
  > "Edward Corser was an Australian politician and land speculator."
  - *Reason / Unsupported Claim*: `Edward Corser engaged in land speculation. [NEUTRAL]`

- **Self-Correcting Action Plan**:
  - **Retries Used**: `1` retry(ies)
  - **Claims Flagged as Unsupported Prior to Retry**:
    - `Edward Corser engaged in land speculation [NEUTRAL]`
- **Self-Correcting Final Answer** (GROUNDED):
  > "Edward Corser was an Australian politician who served in the Queensland Legislative Assembly and the Australian House of Representatives."

---

## Case Study 2: Query #1

**Question**: `What is Herlyn Espinal's occupation?`

- **Plain RAG Answer** (UNGROUNDED):
  > "Herlyn Espinal was a Honduran journalist and government press secretary."
  - *Reason / Unsupported Claim*: `Herlyn Espinal was a government press secretary. [CONTRADICTION]`

- **Self-Correcting Action Plan**:
  - **Retries Used**: `1` retry(ies)
  - **Claims Flagged as Unsupported Prior to Retry**:
    - `Herlyn Espinal served as a government press secretary [CONTRADICTION]`
- **Self-Correcting Final Answer** (GROUNDED):
  > "Herlyn Espinal was a Honduran journalist and television reporter who served as chief correspondent for Televicentro's daily newscast Hoy Mismo."

---

## Case Study 3: Query #7

**Question**: `What is John Finlay's occupation?`

- **Plain RAG Answer** (UNGROUNDED):
  > "John Finlay was an English professional athlete and world champion boxer."
  - *Reason / Unsupported Claim*: `John Finlay was a world champion boxer. [NEUTRAL]`

- **Self-Correcting Action Plan**:
  - **Retries Used**: `2` retry(ies)
  - **Claims Flagged as Unsupported Prior to Retry**:
    - `John Finlay was a world champion boxer [NEUTRAL]`
- **Self-Correcting Final Answer** (GROUNDED):
  > "John Finlay was an English professional footballer who played as an inside-forward for Sunderland AFC."

---

## Case Study 4: Query #8

**Question**: `What is Bruce McDaniel's occupation?`

- **Plain RAG Answer** (UNGROUNDED):
  > "Bruce McDaniel is an American musician, film actor, and award-winning director."
  - *Reason / Unsupported Claim*: `Bruce McDaniel was a film actor and award-winning director. [NEUTRAL]`

- **Self-Correcting Action Plan**:
  - **Retries Used**: `2` retry(ies)
  - **Claims Flagged as Unsupported Prior to Retry**:
    - `Bruce McDaniel was a film actor and director [NEUTRAL]`
- **Self-Correcting Final Answer** (GROUNDED):
  > "Bruce McDaniel is an American musician who works as a composer, music producer, and recording engineer."

---
