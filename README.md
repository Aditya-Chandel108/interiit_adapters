# LoRA Adapters Demo — Domain Router (GPT-2, QLoRA)

A small, runnable Colab notebook that trains two independent **LoRA adapters** on
top of a 4-bit quantized `gpt2`, one per "domain" (contract law, tariffs), routes
incoming queries to the right adapter with a semantic embedding router, and then
evaluates whether each adapter actually **learned** its domain or just
**memorized** its training data.


---

## 1. What the notebook does, cell by cell

**Load base model** — `gpt2`, quantized to 4-bit (`nf4`) via
   `BitsAndBytesConfig`, wrapped in `load_base_model()` so a fresh copy can be
   pulled for each adapter.
**LoRA setup** — `make_lora_model()` wraps a base model with a `LoraConfig`
   (`r=16`, `alpha=32`, targeting GPT-2's `c_attn`/`c_proj` layers) via
   `peft.get_peft_model`. ~1.3M trainable params out of ~126M (1.29%).
**`train_adapter()` helper** — wraps `datasets.Dataset` + `trl.SFTConfig` +
   `trl.SFTTrainer` into one reusable training call.
**Train `q1_contract`** — 15 contract-law Q&A pairs (breach penalties,
   termination, liability caps, indemnification, force majeure, etc.). Adapter
   saved to `./adapters/q1_contract_adapter`.
**Train `q4_tariff`** — a **fresh** base model + fresh LoRA wrapper, then 15
   tariff/customs Q&A pairs (import duties, HS-code misclassification, FTA
   rates, de minimis exemptions, etc.). Adapter saved to
   `./adapters/q4_tariff_adapter`. Trained on a separate model copy so it
   cannot overwrite or blend with the Q1 adapter's weights.
**Semantic router** — reloads the base model once, attaches both adapters by
   name via `peft.PeftModel`, embeds a one-line description of each domain with
   `all-MiniLM-L6-v2`, and routes each query to whichever domain description is
   closest by cosine similarity, then calls `model.set_adapter(...)` before
   generating.
**Evaluation** — see below.

---


## 2. Evaluation methodology (current)

### 2.1 Cross-adapter isolation check
Forces the **wrong** adapter onto each domain's question. Confirms the two
LoRA adapters actually diverged (correct-adapter score is high, wrong-adapter
score is low) rather than both just regurgitating whatever GPT-2 already knew
from pretraining.

### 2.2 Router check
Calls the actual `generate_routed_response()` path — the semantic router picks
an adapter by embedding similarity, and we check both (a) it picked the
correct domain, and (b) the resulting generation contains the expected
keyword. This is the only check that tests the system,as a
user would actually use it.

### 2.3 Memorization vs. generalization check
This is the main evaluation

- **Memorization check** — 2 questions per domain, copied verbatim from the
  15 training examples. Pass condition: one exact expected string appears
  (`"30-day"`, `"12%"`, etc.). A high score here only proves the adapter can
  recite what it was shown — it says nothing about generalization.
- **Generalization check** — 3 *novel* questions per domain, deliberately
  worded differently from anything in the training set (e.g. "Is there a way
  to avoid paying import taxes on low-value packages?" rather than any
  trained phrasing). Since there's no single memorized fact to recite, the
  pass condition is soft: does *any* word from a small domain-vocabulary list
  show up (`duty` / `tariff` / `customs` / `%` for Q4; `cure period` /
  `breach` / `clause` / `notice` for Q1)? A hit here means the adapter carried
  the domain's register into a question it never saw — evidence of learning,
  not just recall.
- **Verdict heuristic** — compares the average memorization vs. average
  generalization score:
  - recall ≥ 80% and generalization < 50% → **memorization**
  - generalization ≥ 60% → **learning**
  - otherwise → **mixed**


---
### 3. Assumptions and Limits for now only

1. 15 handwritten Q&A sentences per domain stood in for what would actually be hundreds of pages of ruling text. This assumption is doing a lot of work — real legal text has internal cross-references, exceptions, and multi-clause structure.
2. The generalization check passes if a domain word shows up anywhere in the output. It does not check whether the legal content of the answer is actually right.
3. I assumed the mechanics (adapter isolation, routing, memorization-vs-generalization behavior) would qualitatively transfer to 8B w/o verifying.

---

## 4. Possible next steps

- Replace the keyword-presence generalization check with embedding-similarity
  or LLM-graded scoring for a less brittle signal.
- Replace the embedding-similarity router with a small trained classifier once
  there are more than two domains.
- Grow each domain's dataset further and add generalization questions per domain.
