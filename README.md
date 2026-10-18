# HiLabs Term Mapping Engine: ICD-10-CM + SNOMED CT

Maps an extracted clinical entity (plus its chunk context) to **one ICD-10-CM code and one SNOMED CT concept id**,
predicted independently per system (neither is derived from a crosswalk of the other).

> Demo video: https://drive.google.com/file/d/1idg8b8HSgZEQU8RDUrm7nQneUD4EPfRo/view?usp=sharing | Write-up: `docs/HiLabs_Documentation.pdf` | Weights (also on Google Drive): https://drive.google.com/file/d/1ppAFcSYWBZTRvYiStmGAzcrOL4-1uqbO/view?usp=sharing

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

### 1. What you need

| Item | Requirement |
|---|---|
| OS | Linux (developed and tested on Kaggle notebooks); macOS and Windows are untested |
| Python | 3.10 or newer |
| GPU | **An NVIDIA GPU with CUDA is required in practice** (tested on a Kaggle T4). The dense encoders (SapBERT, BioLORD), the tuned query encoder and the cross-encoder are impractically slow on CPU |
| Disk | a few GB for the base models from the Hugging Face Hub, the embedding cache (`work/emb_cache`) and the checkpoints (`work/ckpt`) |
| Network | needs the Hugging Face Hub (base models and weights). Mistral's API is needed only if you use the optional LLM feature (see step 4) |
| Data | the three challenge files (see step 3). They are proprietary and are **not** in this repository |

### 2. Install

```bash
git clone <this-repository-url> hilabs-term-mapping
cd hilabs-term-mapping

python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

`requirements.txt` installs `torch` from PyPI. If you need a specific CUDA build, install PyTorch for your CUDA version first
(see pytorch.org), then run the `pip install -r requirements.txt` line. Check that the GPU is visible before going further:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

### 3. Put the data in `data/`

The three files are git-ignored (the data is proprietary and must never be pushed). Copy them into `data/` with these names:

```
data/hackathon_train.csv                          labelled training rows
data/icd10cm_codes.xlsx                           ICD-10-CM dictionary (the legal answer space)
data/SNOMEDCT_US_codes_and_descriptions.xlsx      SNOMED CT dictionary
```

The engine looks for the files by name inside `data/` (the CSV by "train" or "hackathon" plus `.csv`, the dictionaries by "icd" or "snomed"
plus `.xlsx`). If they live elsewhere, set `HILABS_DATA_DIR` or pass `--data-dir /path/to/folder`. If a file is missing, the run stops
immediately with a message that names the missing file.

The file you want predictions for (`--input`) is a CSV with these columns:

| Column | Required | Meaning |
|---|---|---|
| `clinical_entity` (or `entity`) | yes | the extracted diagnosis string |
| `chunk_text` | no, but recommended | the surrounding text; it is used for abbreviations, negation, history and family-history cues |
| `chunk_id` | no | passed through to the output unchanged |

### 4. Configuration (optional): `.env`

Nothing here is required to run. Every line in `.env.example` is commented out, so you can skip this step entirely, or copy it and uncomment what you need:

```bash
cp .env.example .env        # .env is git-ignored and is read automatically by run.py
```

| Variable | Needed? | What it does |
|---|---|---|
| `HF_TOKEN` | no | A Hugging Face **read** token. Not needed, because the base models and the weights repo are public. It only raises anonymous download rate limits. **Leave it unset rather than putting a placeholder in it**: an invalid token makes the Hub reject even public downloads |
| `HF_WEIGHTS_REPO` | no | Overrides the weights repo. The default (`krhk/Hilabs-term-mapping-weights`) is set in `scripts/download_weights.py` |
| `MISTRAL_API_KEY` | no | Turns on the optional LLM pick feature (one small-model call per term, about 240 characters of context). The reported results used it. Without a key the feature is simply empty (see the note below) and **nothing leaves the machine** |
| `HILABS_DATA_DIR`, `HILABS_WORK_DIR`, `HILABS_OUT_DIR` | no | Folders for the input data, caches and checkpoints, and outputs. Defaults: `data/`, `work/`, `outputs/` |
| `USE_LLM_FEATURE` | no | `0` switches the LLM feature off even if a key is present (the same as the `--no-llm` flag) |

**Note on the LLM feature.** The ranker was trained with the LLM pick as one of its inputs. The answers for the training rows are cached in the weights
(`work/llm_feature_cache.json`), but terms in a new evaluation file need a live call. Without a `MISTRAL_API_KEY` those terms get an empty value for that feature, so the ranker still works but
predictions can differ slightly from the reported numbers. Set a key to match the reported setup, or run with `--no-llm` to be sure that no text is sent anywhere.
Keys are read only from the environment or `.env`, and `.env` is git-ignored, so they are never committed.

### 5. Model weights (automatic)

You do not have to do anything. When `run.py predict` or `run.py demo` starts and `work/ckpt` holds no saved checkpoints, it downloads them from the
Hugging Face Hub (see "Model weights" below) into `work/ckpt`. Use `--no-download` to skip that. Without the weights the engine trains the tuned encoders
and the cross-encoders itself, which takes hours on a T4. The notebook's lockbox run, which includes about 2,800 LLM calls, took about 2.6 hours.
A manual download and a Google Drive backup copy are described under "Model weights".

### 6. Check that it works

```bash
python run.py demo
```

This refits the engine on all labelled rows (reusing the downloaded weights, so the encoders are not retrained) and then prints, for ten hard cases, the top candidates of each system with their confidence (for example `HTN` -> `I10`, and the OCR-mangled
`chronickidney disease stag 3` -> `N18.30`). The first lines list the three data files that were found. If you see them and the candidate tables, the setup is correct. Then continue with the
single-command reproduction below.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: Input file not found ...` | A data file is missing from `data/`. Check the three names above, or set `HILABS_DATA_DIR` |
| `torch.cuda.is_available()` is `False`, or very slow, or CUDA out of memory | The run needs a CUDA GPU. Use a machine with a T4 or better, and close other GPU jobs |
| `401` or "invalid token" from Hugging Face | An invalid `HF_TOKEN` is set. Unset it or put a real read token in `.env` |
| `WARNING: weights not downloaded; the engine will train from scratch` | The Hub download failed (offline, or the weights repo has no files yet). Use the Google Drive copy (see "Model weights"), or accept the long first run |
| The run retrains the encoders even though weights exist | Checkpoint names are hashes of the training data and settings. They only match if the data files are the original ones and the zip came from the fit-on-all-rows run (notebook section 15b) |
| `LLM feature error ...` or "picks for x/y terms" in the log | A Mistral request failed or was rate limited. The run continues and treats the missing answers as empty. Re-running reuses every cached answer |
| `predicted codes outside the dictionary` | The engine checks every output against the supplied dictionaries. This means the dictionary files are not the original ones |

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
python run.py demo        # refits on all labelled rows, then prints ten hard cases with top candidates and confidence
```

## Model weights (Hugging Face)

Fine-tuned weights (tuned SapBERT query encoders, MedCPT cross-encoders, cached LLM answers) are hosted on the Hugging Face Hub:

**[`krhk/Hilabs-term-mapping-weights`](https://huggingface.co/krhk/Hilabs-term-mapping-weights)**  (public)

No setup needed: `python run.py predict ...` downloads them into `work/ckpt` automatically when none are present (the repo id is set in `scripts/download_weights.py`; no token is needed because the repo is public), or do it by hand: `python scripts/download_weights.py <hf_user>/<repo>`.
Maintainers upload them with `python scripts/export_to_hf.py`. The base models (SapBERT, BioLORD-2023, MedCPT) are pulled from their own public HF repos at run time.

**Backup copy.** The same weights are also available as a single zip on Google Drive: [ADD DRIVE LINK TO weights.zip]. If the Hugging Face download is unavailable, download it, then run `python scripts/restore_weights.py weights.zip` (use the file's actual name) and re-run the command above. This unpacks the checkpoints into `work/`.

Tokens are read from the environment only and are never committed.

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
scripts/download_weights.py pull checkpoints from the HF Hub
scripts/restore_weights.py unpack checkpoints from a local zip
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
