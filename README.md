# VK---HiLabs-AI-Quest-Hackathon-Clinical-Track-
Clinical Mapping Engine for ICD-10 and SNOMED.

Maps an extracted clinical entity (plus its chunk context) to **one ICD-10-CM code and one SNOMED CT concept id**,
predicted independently per system (neither is derived from a crosswalk of the other).

> Demo video: `[ADD INTERNAL DRIVE LINK]` | Write-up: `docs/HiLabs_Documentation.pdf`

## Pipeline

| Stage | What it does |
|---|---|
| Text preparation | normalisation, spelling repair against the dictionary vocabulary (corrections are *appended*, never replace), abbreviation expansion (static list + definitions found in the chunk, e.g. `chronic kidney disease (CKD)`), lay-term expansion, measurement/grade stripping, negation / history / family-history cues |
| Retrieval (per system) | TF-IDF word + char n-grams, SapBERT, BioLORD-2023, a SapBERT *query* encoder fine-tuned on the training pairs (cross-fitted), a chunk-context channel, and the training memory (training terms act as extra synonyms of their gold code, chunk-grouped leave-out) |
| ICD hierarchy | for every parent/category candidate, the default billable child (`R05 -> R05.9`) is injected; sibling and specificity features |
| Cross-encoder | `ncbi/MedCPT-Cross-Encoder` fine-tuned on our labels (cross-fitted); candidate text = the code's descriptions most similar to the term |
| LLM pick feature (optional) | one `ministral-8b` call per term picks among each system's top-5; LightGBM learns when to trust it. Temperature 0, fixed seed, answers cached. Missing key = feature is NaN and the ranker works without it |
| Reranker | LightGBM LambdaRank per system; boosting rounds chosen by chunk-grouped CV; the output is the top-1 code plus a softmax confidence |

Predicted codes are always checked against the supplied dictionaries (a code outside them is scored wrong, so the engine asserts none is emitted).

## Setup

Needs Python 3.10+ and a **GPU** (tested on Kaggle T4; the dense encoders, tuned encoder and cross-encoder are impractical on CPU).

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in HF_TOKEN (optional) and MISTRAL_API_KEY (optional)
```

Put the three challenge files in `data/` (they are git-ignored; the data is proprietary and must not be pushed):

```
data/hackathon_train.csv
data/icd10cm_codes.xlsx
data/SNOMEDCT_US_codes_and_descriptions.xlsx
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | no | Hugging Face token for downloading the public models (SapBERT, BioLORD-2023, MedCPT) without anonymous rate limits |
| `MISTRAL_API_KEY` | no | Enables the LLM pick feature. Without it (or with `--no-llm`) nothing leaves the machine |
| `HILABS_DATA_DIR` / `HILABS_WORK_DIR` / `HILABS_OUT_DIR` | no | defaults `data/`, `work/`, `outputs/` |

## Reproduce the submitted predictions (single command)

```bash
python run.py predict --input data/evaluation_input.csv --output outputs/predictions.csv
```

Input needs `clinical_entity` (or `entity`); `chunk_text` and `chunk_id` are used/passed through when present.
Output columns: `chunk_id, clinical_entity, icd10cm_code, snomed_code`. No manual step is involved.

**Fast path with the saved weights.** The first run trains the tuned encoders and cross-encoders and fills the
embedding cache (hours on a T4). To skip that, unpack the checkpoints exported by notebook section 15c first:

```bash
python scripts/restore_weights.py hilabs_weights.zip      # copies tuned/cross-encoder checkpoints + cached LLM answers into work/
python run.py predict --input data/evaluation_input.csv --output outputs/predictions.csv
```

Checkpoint names are hashes of the training data and settings, so they are only reused if the data/settings match
(use the zip from a run fitted on all labelled rows, notebook section 15b with `FIT_ALL = True`).

### Other commands

```bash
python run.py lockbox     # fit on train+dev folds, score the held-out lockbox once; writes outputs/test_predictions.csv + test_errors.csv
python run.py demo        # hard-case demo with top candidates and confidence
python scripts/make_report.py --out docs/HiLabs_Documentation.pdf   # regenerate the write-up (fills numbers from outputs/ if present)
```

## Determinism

Seed 42 everywhere (split, tuner, cross-encoder, LightGBM `deterministic=True`). The only external call is the Mistral
pick feature: temperature 0, fixed `random_seed`, every answer cached on disk (`work/llm_feature_cache.json`), so re-runs
are identical and free.

## Evaluation protocol

The labelled file is split by chunk (groups sorted, shuffled with seed 42): 15% of chunks form a **lockbox**, the rest
are dealt into 5 CV folds. Fold 4 is dev (used only to decide whether the cross-encoder feature helps, per system).
The lockbox is scored once, by a model refitted on all CV folds (`python run.py lockbox`). The submission model is then
refitted on every labelled row.

## Repository layout

```
run.py                     CLI entry point (predict | lockbox | demo)
src/engine.py              the engine (auto-generated from the notebook, see header)
notebooks/hilabs_final.ipynb   source of truth: research cells, summary tables, diagnostics, appendix
scripts/restore_weights.py unpack saved checkpoints
scripts/make_report.py     builds the documentation PDF
docs/                      write-up PDF + demo script
```

`src/engine.py` is regenerated from the notebook; edit the notebook, not the module.

## External resources (all open, named for the write-up)

| Resource | Use | Licence |
|---|---|---|
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | dense retrieval; query encoder fine-tuned on our labels | check model card |
| `FremyCompany/BioLORD-2023` | second dense channel | check model card |
| `ncbi/MedCPT-Cross-Encoder` | cross-encoder, fine-tuned on our labels | check model card |
| `ministral-8b-2512` via Mistral La Plateforme | optional pick feature | Mistral API terms |
| LightGBM, scikit-learn, rapidfuzz, ftfy | ranking, TF-IDF, spelling repair, text fixing | open source |
| ICD-10-CM and SNOMED CT dictionaries | supplied by the organisers (legal answer space) | as supplied |

Not used: UMLS and the official SNOMED-to-ICD-10-CM map (the hook exists as `XMAP_PATH`, left unset because it needs a UMLS licence).
Confirm the MedCPT and Mistral licence terms before submission and fix the table above if they differ.

## Data handling

The challenge data is proprietary: it is git-ignored, never committed, and the repository must stay **private**.
With `MISTRAL_API_KEY` set, the term and ~240 characters of chunk context per term are sent to Mistral's hosted API for
the optional pick feature. If that is not acceptable for the data, run with `--no-llm`. Self-hosting an open-weight
model for this step is the clean alternative.
