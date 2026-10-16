# AI Platform Engineer Scenario Test

Two small, self-contained prototypes plus their production architecture:

| Scenario | Notebook | What it does |
|---|---|---|
| **1. Customer Support Classifier** | `Scenario_1_Customer_Support_Classifier.ipynb` | Classifies support tickets into `billing`, `technical_issue`, or `general_inquiry` using TF-IDF + Logistic Regression |
| **2. Engineering Support Chatbot** | `Scenario_2_Chatbot.ipynb` | Retrieval-grounded chatbot over an engineering knowledge base, using Gemini with hard and soft hallucination guardrails |

Both notebooks were written for **Google Colab** but run locally with minor path changes (see [Setup](#setup)).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Setup](#setup)
3. [Running the Notebooks](#running-the-notebooks)
4. [Architecture: Scenario 1 (Classifier)](#architecture-scenario-1--customer-support-classifier)
5. [Architecture: Scenario 2 (Chatbot)](#architecture-scenario-2--engineering-support-chatbot)
6. [Known Limitations & Next Steps](#known-limitations--next-steps)

---

## Repository Layout

```
.
├── README.md
├── Scenario_1_Customer_Support_Classifier.ipynb
├── Scenario_2_Chatbot.ipynb
│
│   # Generated when the notebooks run
├── customer_support_db.json        # Scenario 1: 13 labeled dummy tickets
├── eng_knowledge_base.json         # Scenario 2: 8 dummy KB articles (Colab: /content/)
└── .env                            # Scenario 2: GOOGLE_API_KEY (local runs only, never commit)
```

---

## Setup

### Prerequisites

- Python 3.9+
- A **Google Gemini API key** (Scenario 2 only). Create one at <https://aistudio.google.com/apikey>.
- Scenario 1 needs no API key and no network access beyond installing packages.

### Option A: Google Colab (recommended, zero install)

1. Upload both `.ipynb` files to [Colab](https://colab.research.google.com/) (File → Upload notebook).
2. **Scenario 2 only:** add your API key as a Colab Secret.
   - Click the 🔑 **Secrets** icon in the left sidebar.
   - Add a secret named `GOOGLE_API_KEY` with your key as the value.
   - Toggle **Notebook access** on.
3. Run cells top to bottom (Runtime → Run all).

`pandas`, `scikit-learn` and `google-genai` are available or installable in Colab (`!pip install -q google-genai` if the import fails).

### Option B: Local machine

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install pandas scikit-learn google-genai jupyter

# 3. (Scenario 2 only) provide your API key, using ONE of:
export GOOGLE_API_KEY="your-key-here"                 # environment variable
# or
echo "GOOGLE_API_KEY='your-key-here'" > .env          # local .env file

# 4. Launch Jupyter
jupyter notebook
```

**Local-run adjustments for Scenario 2:**

- The notebook hard-codes Colab paths (`/content/eng_knowledge_base.json`). Replace `/content/` with `./` in the KB-creation cells and in `retrieve_context(..., kb_path=...)`.
- The first `get_api_key()` helper and its import (`from google.colab import userdata`) sit at the top of the cell and will raise `ModuleNotFoundError` outside Colab. Either delete that import or use the later, corrected version of `get_api_key()` in the "Additional validation" cell, which wraps the import in `try/except`.

### API key resolution order (Scenario 2)

The chatbot looks for `GOOGLE_API_KEY` in this order and uses the first match:

1. Colab Secrets (`google.colab.userdata`)
2. Environment variable
3. A local `.env` file (`GOOGLE_API_KEY='...'`)

> ⚠️ **Never commit `.env` or paste your key into a notebook cell.** Add `.env` to `.gitignore`.
> The notebook includes a helper cell that copies a key found in Colab Secrets or the environment into a local `.env`. This is a convenience for demos; remove it for anything shared.

### Model name

Scenario 2 calls Gemini with `model='gemini-3.5-flash-lite'`. If your account or SDK version does not expose that model, swap in any available Gemini Flash-class model; no other code changes are needed.

---

## Running the Notebooks

### Scenario 1: Classifier

Run cells in order:

1. **Generate data:** writes `customer_support_db.json` (13 tickets, 3 labels).
2. **Load and split:** 70/30 stratified train/test split (`random_state=42`).
3. **Train and evaluate:** fits the pipeline and prints a `classification_report` and overall accuracy.
4. **Inference tests:** classifies 6 unseen messages (3 of them are re-run as edge cases) and prints the predicted category.

### Scenario 2: Chatbot

Run cells in order:

1. **Build the knowledge base:** creates `eng_knowledge_base.json`, then expands it (6 entries, then 8 with Kafka and React).
2. **Define `retrieve_context`, `get_api_key`, `engineer_chatbot`, `engineer_chatbot_v2`.** Use **v2** (few-shot prompt).
3. **Run the test queries.** In-scope queries return a grounded answer; out-of-scope queries (e.g. "How do I write a Python loop?") hit the guardrail and return a safe refusal **without calling the LLM**.

---

## Architecture: Scenario 1 – Customer Support Classifier

### Goal

Route each incoming support message to the right team by predicting one of three categories.

### What is implemented (prototype)

```
 customer_support_db.json ──► train/test split (70/30, stratified)
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │ sklearn Pipeline                 │
                    │  1. TfidfVectorizer              │
                    │     - English stop words         │
                    │     - unigrams + bigrams (1,2)   │
                    │  2. LogisticRegression           │
                    └──────────────────────────────────┘
                                       │
                                       ▼
                       predicted category + report
```

**Why this baseline:** fast to train, explainable through per-word coefficients, and a sensible reference point before moving to heavier models.

### Production blueprint (design)

```
  [ Client Apps / Widgets / Web Portal ]
                   │  POST /v1/tickets/classify
                   ▼
         [ API Gateway Router ]
                   │
                   ├─────────────────────────────────────────┐
                   ▼ (auth & rate limit)                     ▼ (async log)
        [ Serverless ML Gateway ]                    [ System DB / Event Log ]
         (GCP Cloud Run / AWS Lambda)
                   │
                   ▼ Text preprocessor
                   ▼ TF-IDF / embedding extraction
                   ▼ Category prediction
                   │
                   ├──► "billing"          ──► Stripe/Zuora integration ──► Billing queue
                   ├──► "technical_issue"  ──► Datadog / DevOps alert   ──► Engineering SRE
                   └──► "general_inquiry"  ──► Auto-response draft      ──► Support pool
```

| Layer | Responsibility |
|---|---|
| **API Gateway** | Single entry point for web, mobile, and helpdesk integrations (e.g. Zendesk); handles auth and rate limiting; logs asynchronously |
| **Inference service** | Stateless serverless function: clean → vectorize → predict |
| **Routing broker** | Maps each class to a downstream action and queue |

### Design deliverables

**Data needed**
- Labeled historical tickets (verified categories)
- Metadata: timestamp, channel (web/app/API/email), customer tier, conversation depth
- Varied phrasing: multi-sentence, fragments, jargon, typos, mixed sentiment, multiple languages

**Preprocessing**
- Normalization: lowercase, strip HTML and URLs, anonymize PII
- Tokenization and stop-word removal
- Lemmatization (crashed/crashing → crash)
- Vectorization: TF-IDF baseline → dense embeddings (e.g. `text-embedding-3-small`, SentenceTransformers) for semantic context

**Model choices**

| Tier | Model | Why |
|---|---|---|
| Baseline (implemented) | TF-IDF + Logistic Regression | Fast, cheap, interpretable |
| Production option A | Fine-tuned DistilBERT | Self-attention captures context ("the balance won't load" is a technical issue, not billing) |
| Production option B | LLM with few-shot prompting | Strong out of the box, minimal infrastructure |

**Evaluation**
- Offline: macro F1 (handles class imbalance), precision, recall
- Shadow deployments / A-B tests on live traffic before cutover
- Online: track agent manual overrides as an error signal

**Risks and mitigations**

| Risk | Mitigation |
|---|---|
| Concept drift (new products, new issues) | Scheduled retraining on freshly labeled data |
| Class imbalance | Class-weighted loss, SMOTE / synthetic data |
| Sarcasm and complex syntax | Secondary sentiment check; escalate low-confidence tickets to humans |

---

## Architecture: Scenario 2 – Engineering Support Chatbot

### Goal

Answer engineers' questions **only** from a trusted internal knowledge base, and refuse rather than guess when the answer is not there.

### End-to-end flow

```
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ Keyword / Match Retriever│
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
     [overlap ≥ 2 words]                      [overlap < 2 words]
              │                                       │
              ▼                                       ▼
   ┌──────────────────────┐              ┌──────────────────────────┐
   │ Fetch KB doc context │              │ HARD GUARDRAIL           │
   └──────────┬───────────┘              │ Return safe refusal;     │
              ▼                          │ LLM is never called      │
   ┌──────────────────────┐              └──────────────────────────┘
   │ Few-shot structured  │
   │ prompt (system +     │
   │ context + question)  │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Gemini API inference │   ◄── SOFT GUARDRAIL (grounding instructions)
   └──────────┬───────────┘
              ▼
        Grounded response
```

### Components

| Component | Implementation | Notes |
|---|---|---|
| **Knowledge base** | `eng_knowledge_base.json`: entries with `id`, `topic`, `category`, `content` | 8 articles: PostgreSQL timeouts, CI/CD OOM, Redis eviction, API rate limits, Docker DNS, Elasticsearch yellow status, Kafka consumer lag, React hydration |
| **Retriever** | `retrieve_context(query)`: lower-cases and whitespace-splits the query, scores each entry by word overlap with `topic + content`, returns the best match | Requires **≥ 2** overlapping words, otherwise returns `None` |
| **Hard guardrail** | If retrieval returns `None`, return a fixed refusal string | Deterministic; skips the LLM call so no hallucination and no token cost |
| **Prompt** | System instruction + two few-shot examples (one *answerable*, one *unanswerable*) + `Context / Question / Answer` | Teaches both the answer style and the refusal style |
| **LLM** | Gemini via the `google-genai` SDK (`client.models.generate_content`) | System prompt passed through `config={'system_instruction': ...}` |
| **Secrets** | `get_api_key()` | Colab Secrets → env var → `.env` |
| **Error handling** | Separate `try/except` around client init and generation | Returns readable error strings instead of crashing |

### Prompt design

1. **Role:** "expert technical support assistant for software engineers."
2. **Constraint:** answer using **only** the supplied context; do not extrapolate, assume, or invent.
3. **Few-shot examples:**
   - *Match:* context and question align → answer, citing the documentation.
   - *No match:* context is related but insufficient → state that there is not enough verified information, and say what the documentation *does* cover.
4. **User turn:** `Context: … Question: … Answer:`

### Hallucination guardrails (defense in depth)

| Layer | Type | Mechanism |
|---|---|---|
| 1 | **Deterministic (hard)** | Python-side overlap threshold blocks out-of-scope queries before any LLM call |
| 2 | **Generative (soft)** | System instruction and few-shot refusal example keep the model within the retrieved context |
| 3 | **Operational** | Fixed, user-friendly refusal text pointing to a tech lead or internal docs |

### Example behavior

| Query | Result |
|---|---|
| "How do I fix container DNS issues?" | Retrieves `kb_005` and answers with the `daemon.json` DNS fix |
| "What should I do if Elasticsearch status goes yellow?" | Retrieves `kb_006` and answers from it |
| "How do I write a Python loop?" | No KB match → hard guardrail refusal (no LLM call) |

---

## Known Limitations & Next Steps

These are honest gaps between the prototypes and a production system.

**Scenario 1**
- The dataset is tiny (13 tickets), so metrics from the baseline are illustrative, not representative. A real evaluation needs hundreds or thousands of labeled tickets per class.
- The preprocessing described in the design deliverables (PII scrubbing, lemmatization, embeddings) is documented but **not** implemented; the code relies on `TfidfVectorizer`'s built-in tokenization and stop-word removal.
- The API gateway, routing broker, and downstream integrations are architectural design only.
- Ambiguous messages (e.g. "refund policy", "close my account due to charges") can sit between categories; consider confidence thresholds with human fallback.

**Scenario 2**
- **Retrieval is naive.** Whitespace tokenization with no punctuation stripping, stemming, or synonym handling means paraphrased queries can miss. Only the single best document is returned.
- **Upgrade path:** embedding-based retrieval (e.g. sentence embeddings + a vector store such as FAISS, pgvector, or Chroma), top-k retrieval, optional re-ranking, and source citations in answers.
- The threshold of 2 overlapping words is a heuristic and should be tuned on a labeled query set.
- Common stop words ("the", "to", "in") count toward overlap, so generic questions can pass the guardrail. Remove stop words before scoring.
- No conversation memory, logging, or evaluation harness yet. Add a test set of in-scope and out-of-scope questions to measure refusal accuracy and groundedness.
- The notebook defines `get_api_key()` and the chatbot functions more than once; consolidating them into a single module would be cleaner.
