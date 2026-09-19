# LoRA Adapters Demo — Domain Router (GPT-2, QLoRA)

A small, runnable Colab notebook that trains two independent **LoRA adapters** on
top of a 4-bit quantized `gpt2`, one per "domain" (contract law, tariffs), routes
incoming queries to the right adapter with a semantic embedding router, and then
evaluates whether each adapter actually **learned** its domain or just
**memorized** its training data.

Runtime: Colab, T4 GPU.

---

## 1. What the notebook does, cell by cell

1. **Install deps** — `transformers`, `torch`, `huggingface_hub`, `bitsandbytes`,
   `trl`, `peft`, `datasets`, `accelerate`, `sentence-transformers`.
2. **(Optional) HF login** — only needed if you swap `model_id` for a gated model
   like Llama; not required for the default `gpt2`.
3. **Load base model** — `gpt2`, quantized to 4-bit (`nf4`) via
   `BitsAndBytesConfig`, wrapped in `load_base_model()` so a fresh copy can be
   pulled for each adapter.
4. **LoRA setup** — `make_lora_model()` wraps a base model with a `LoraConfig`
   (`r=16`, `alpha=32`, targeting GPT-2's `c_attn`/`c_proj` layers) via
   `peft.get_peft_model`. ~1.3M trainable params out of ~126M (1.29%).
5. **`train_adapter()` helper** — wraps `datasets.Dataset` + `trl.SFTConfig` +
   `trl.SFTTrainer` into one reusable training call.
6. **Train `q1_contract`** — 15 contract-law Q&A pairs (breach penalties,
   termination, liability caps, indemnification, force majeure, etc.). Adapter
   saved to `./adapters/q1_contract_adapter`.
7. **Train `q4_tariff`** — a **fresh** base model + fresh LoRA wrapper, then 15
   tariff/customs Q&A pairs (import duties, HS-code misclassification, FTA
   rates, de minimis exemptions, etc.). Adapter saved to
   `./adapters/q4_tariff_adapter`. Trained on a separate model copy so it
   cannot overwrite or blend with the Q1 adapter's weights.
8. **Semantic router** — reloads the base model once, attaches both adapters by
   name via `peft.PeftModel`, embeds a one-line description of each domain with
   `all-MiniLM-L6-v2`, and routes each query to whichever domain description is
   closest by cosine similarity, then calls `model.set_adapter(...)` before
   generating.
9. **Evaluation** — see §3 below.

---

## 2. Bugs found and fixed (in the order they came up)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Install cell silently no-ops | Missing `!` before `pip install`, so it's not a shell command | Added `!`, merged all installs into one cell, added the packages that were imported but never installed (`peft`, `datasets`, `sentence-transformers`, `accelerate`) |
| 2 | `AttributeError: 'list' object has no attribute 'endswith'` in `trainer.train()` | `formatting_func` returned `[text]` (a list); `SFTTrainer` calls it **per example**, not per batch, and expects a plain string | `format_prompts()` now returns a single string |
| 3 | Router cell fails to load `./adapters/q4_tariff_adapter` | That adapter was never trained — only `q1_contract` had a training cell | Added a full training cell for `q4_tariff`, using a fresh base model so it trains independently of `q1_contract` |
| 4 | Deprecation warning / brittle reload | `load_in_4bit=True` passed directly to `from_pretrained` at reload time | Replaced with `quantization_config=bnb_config` (matches the initial load) |
| 5 | `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` during training | `fp16=True` mixed-precision uses a `GradScaler` that only supports float16 gradients, but some tensors ended up bfloat16 under 4-bit quantization | Set `fp16=False`, `bf16=False` — train the (tiny) LoRA params in plain fp32; the frozen base model stays 4-bit regardless |
| 6 | Generation loops the same sentence 6–8 times verbatim (`"12% customs duty applies to cross-border electronics shipments."` repeated) | Severe overfitting from `max_steps=100` on a **single** training example per domain, plus no repetition control at generation time | Added `repetition_penalty=1.3`, `no_repeat_ngram_size=3` to all `generate()` calls; later, expanded training data from 1 → 15 examples per domain so the adapter has more than one string to memorize |
| 7 | First eval run reported 100% / 100% and looked suspicious | `evaluate_domain_accuracy` called `model.set_adapter(adapter_name)` **directly**, bypassing the router — it never tested whether routing worked, only whether a manually-forced adapter could recite its (single, heavily overfit) training fact | Rewrote eval into three checks: forced-adapter recall, cross-adapter isolation (force the *wrong* adapter — should fail), and true end-to-end router accuracy (see §3) |
| 8 | Still unclear whether the adapters learned the domain or just memorized one fact, even after fix #7 | 1 training example ⇒ can't distinguish memorization from learning; there's only one fact to test | Expanded each domain to 15 varied examples and split eval into an explicit **memorization check** (verbatim training questions) vs. **generalization check** (novel, differently-phrased questions, scored by domain-vocabulary presence, not one exact string) |

---

## 3. Evaluation methodology (current)

### 3.1 Cross-adapter isolation check
Forces the **wrong** adapter onto each domain's question. Confirms the two
LoRA adapters actually diverged (correct-adapter score is high, wrong-adapter
score is low) rather than both just regurgitating whatever GPT-2 already knew
from pretraining.

### 3.2 Router check
Calls the actual `generate_routed_response()` path — the semantic router picks
an adapter by embedding similarity, and we check both (a) it picked the
correct domain, and (b) the resulting generation contains the expected
keyword. This is the only check that tests the system **end-to-end**, as a
user would actually use it.

### 3.3 Memorization vs. generalization check
This is the main evaluation and the one most recently added:

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

All generations are printed raw (not just pass/fail) so you can visually sanity-check the model's actual output, not just trust the keyword match.

### 3.4 What this does *not* do (known limitations)
- No held-out validation loss from training itself — evaluation is entirely
  post-hoc, on hand-written test questions.
- The generalization "pass" condition is a soft keyword-presence check, not a
  semantic or human-graded correctness judgment.
- `gpt2` (124M params) and 15 examples/domain is a toy scale — good for
  demonstrating the *mechanics* of adapter isolation, routing, and
  memorization-vs-generalization testing, not a rigorous accuracy benchmark.
- No true stability/plasticity (continual-learning) test — see next section.

---

## 4. "Stability/plasticity" — why it's not used here

The original notebook borrowed continual-learning terminology
("Historical Retention (Stability)" / "New Knowledge Mastery (Plasticity)"),
which implies a sequential setup: train task A, then continue training the
*same* weights on task B, then check whether B's training damaged A (stability)
while confirming B was actually learned (plasticity).

That framing doesn't fit this notebook's architecture. Since the fix that
added Q4 adapter training, **each domain gets its own adapter trained from an
independent copy of the frozen base model** — they never share weights and
can't overwrite each other by construction (confirmed by the cross-adapter
isolation check in §3.1). There is no forgetting risk to measure, so a
"plasticity score" here would just be relabeling "adapter 2's recall
accuracy," which is why the eval was renamed to memorization/generalization —
terms that describe what's actually being measured.

A genuine stability/plasticity test would require a different experiment:
train one adapter on Q1, keep training that *same* adapter on Q4, then check
whether Q1 accuracy dropped (stability) while Q4 accuracy rose (plasticity).
That's a valid and interesting addition but is architecturally different from
the multi-adapter routing setup this notebook currently implements.

---

## 5. Possible next steps

- Add the true continual-learning stability/plasticity test described above.
- Replace the keyword-presence generalization check with embedding-similarity
  or LLM-graded scoring for a less brittle signal.
- Swap `gpt2` for a larger base model (the notebook already supports gated
  models like Llama via the commented-out `model_id`).
- Replace the embedding-similarity router with a small trained classifier once
  there are more than two domains.
- Grow each domain's dataset further and add more held-out generalization
  questions per domain for a statistically sturdier score (currently n=2–3
  per check, which is still small).
