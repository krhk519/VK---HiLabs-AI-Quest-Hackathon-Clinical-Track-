"""HiLabs term-mapping engine: clinical entity (+ chunk context) -> ICD-10-CM + SNOMED CT.

AUTO-GENERATED from notebooks/hilabs_final.ipynb (the code cells, in order, with the Kaggle/Colab path handling
replaced by environment variables). Importing this module loads the data, builds the dictionaries and defines the
engine classes; `run.py` is the command-line entry point. Do not edit by hand: edit the notebook and regenerate.
"""
import os
try:                                           # optional: read keys from a local .env (never committed)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
if os.environ.get("HF_TOKEN"):                 # Hugging Face token (optional: raises download rate limits)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


# ====================================================================================================

import os, re, json, math, time, hashlib, random, gc, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# ============================== PATHS (environment driven) ==============================
# HILABS_DATA_DIR : folder with the 3 input files (default ./data)
# HILABS_WORK_DIR : caches + checkpoints (default ./work)      HILABS_OUT_DIR : outputs (default ./outputs)
IN_KAGGLE = os.path.isdir("/kaggle/input")
DATA_DIR = os.environ.get("HILABS_DATA_DIR", "data")

def _find_input(patterns, roots):
    """first file under any root whose lower-cased name matches one of the substring groups (all must match)"""
    for r in roots:
        for dp, _, fns in os.walk(r):
            for fn in sorted(fns):
                l = fn.lower()
                if any(all(p in l for p in pat) for pat in patterns): return os.path.join(dp, fn)
    return None

_ROOTS = [DATA_DIR] + (["/kaggle/input"] if IN_KAGGLE else [])
CSV_PATH    = _find_input([("train", ".csv"), ("hackathon", ".csv")], _ROOTS) or f"{DATA_DIR}/hackathon_train.csv"
ICD_XLSX    = _find_input([("icd", ".xlsx")], _ROOTS) or f"{DATA_DIR}/icd10cm_codes.xlsx"
SNOMED_XLSX = _find_input([("snomed", ".xlsx")], _ROOTS) or f"{DATA_DIR}/SNOMEDCT_US_codes_and_descriptions.xlsx"
for p_ in (CSV_PATH, ICD_XLSX, SNOMED_XLSX):
    if not os.path.exists(p_):
        raise FileNotFoundError(f"Input file not found: {p_}. Put hackathon_train.csv, icd10cm_codes.xlsx and "
                                f"SNOMEDCT_US_codes_and_descriptions.xlsx in {DATA_DIR}/ (or set HILABS_DATA_DIR).")
print("CSV   :", CSV_PATH); print("ICD   :", ICD_XLSX); print("SNOMED:", SNOMED_XLSX)
XMAP_PATH   = None   # optional: NLM "SNOMED CT to ICD-10-CM" map (tab separated, needs a free UMLS licence). None = skip.
WORK_DIR = os.environ.get("HILABS_WORK_DIR", "work")
OUT_DIR  = os.environ.get("HILABS_OUT_DIR", "outputs")
EMB_DIR  = f"{WORK_DIR}/emb_cache"
for d_ in (WORK_DIR, OUT_DIR, EMB_DIR): os.makedirs(d_, exist_ok=True)

# ============================== PIPELINE SWITCHES ==============================
SEED = 42
K = 50                  # candidates kept per query from retrieval
USE_SPELL = True        # fix OCR/typo tokens against the dictionary vocabulary
USE_ABBREV = True       # append expansions of known abbreviations (+ definitions found in the chunk)
USE_DENSE = True        # SapBERT dense retrieval channel (GPU recommended)
ICD_LEAF_FIX = True     # v5: inject the default billable child of every parent/category ICD candidate
SIBLING_FEATS = True    # v6: sibling-aware reranker features (ICD specificity decision)
USE_COUPLING = True     # cross-system features (ICD <-> SNOMED agreement learned from training pairs / optional official map)
SAPBERT_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
K_EXTRA = 40            # candidates kept from each NEW channel (tuned SapBERT, BioLORD)

# ---- improvements added in v2 (all switchable; everything needs the GPU runtime) ----
USE_BIOLORD = True      # 2nd frozen dense channel: BioLORD-2023 (first run encodes ~480k dictionary texts, ~5 min, cached)
BIOLORD_NAME = "FremyCompany/BioLORD-2023"
USE_TUNED = True        # SapBERT fine-tuned on the training pairs (query side), cross-fitted so the reranker sees honest scores
TUNE_EPOCHS, TUNE_LR, TUNE_BS, TUNE_TAU, TUNE_FOLDS = 5, 2e-5, 32, 0.05, 5
TUNE_CTX_CHARS = 0      # 0 = term only (verified). >0 = also feed this many characters of chunk context on each side (experiment)
CE_BASE_NAME = "ncbi/MedCPT-Cross-Encoder"   # v6: pretrained biomedical reranker, fine-tuned on our labels (falls back to SapBERT)
USE_CE = True           # cross-encoder: reads  term + chunk context + candidate description  together; its score is a reranker feature
CE_MODE = "auto"        # "auto": use the cross-encoder per system only if it wins on dev | "on" | "off"
CE_N = 30               # candidates per term that the cross-encoder scores (best by fused retrieval score)
CE_FOLDS, CE_EPOCHS, CE_GROUP, CE_BATCHQ, CE_LR, CE_CTX_CHARS = 4, 2, 8, 8, 3e-5, 160

# ============================== v7 ==============================
USE_LAY = True          # lay words -> clinical words, APPENDED to the query (broke -> fracture, heart attack -> myocardial infarction)
USE_CTX_DENSE = True    # chunk-context channel: SapBERT of the text around the term vs every candidate (feature + extra candidates)
K_CTX, CTX_W, CTX_CHARS = 20, 0.5, 120   # extra candidates from (term + CTX_W x context) | characters of context on each side
USE_BODY = True         # body-region agreement features (tooth vs spine)
USE_CUES = True         # negation / history / family-history cues: reranker features + cue-aware search
K_CUE = 15              # extra candidates from the cue-aware search ("history of <term>")

# ============================== FINAL: evaluation split ==============================
SPLIT = "lockbox"       # "lockbox" = research notebook folds (train = CV folds - DEV_FOLD, dev = DEV_FOLD, test = lockbox)
                        # "random"  = the earlier 70/15/15 GroupShuffleSplit
DEV_FOLD = "4"
EXPECTED_ICD_PER_FOLD = {"0": 375, "1": 393, "2": 393, "3": 394, "4": 389, "lockbox": 362}   # research notebook, mentions with ICD gold

# ============================== v8 (merged with the teammate notebook) ==============================
GROUP_LOO = True        # training rows never see train-memory rows / priors / pairs from their OWN CHUNK (not only themselves)
USE_QTY_STRIP = True    # remove measurements and grades from the query ("2.9 cm", "+2", "4/5", "100%")
CE_DESC = "mention"     # cross-encoder candidate text: "mention" = the code's descriptions most similar to the term | "title" (v7)
CE_DESC_MAX = 180       # characters of candidate text for the cross-encoder
RANKER_ROUNDS = (100, 200, 300, 400, 600)   # LightGBM boosting rounds tried; the best is chosen by chunk-grouped CV on training rows
RANKER_CV = 3           # folds for that choice (() or 0 = fixed 300 rounds as in v7)
USE_LLM_FEATURE = os.environ.get("USE_LLM_FEATURE", "1") == "1"   # needs MISTRAL_API_KEY; off if the key is missing
FEATURE_PROVIDER = "mistral"            # teammate setup: Mistral free tier
FEATURE_MODEL = "ministral-8b-2512"     # cheap model, ~3 requests/s on the free tier
FEATURE_RPS = 2.5                        # requests per second (account limit for ministral-8b: 3.13)
FEATURE_TOP_N = 5                        # candidates per system shown to the LLM
FEATURE_EX = 2                           # similar training terms (other chunks only) shown as examples, per system
FEATURE_CACHE = f"{WORK_DIR}/llm_feature_cache.json"
FEATURE_CALL_OVERRIDE = None             # testing hook: a function(prompt) -> answer text

random.seed(SEED); np.random.seed(SEED)
T0 = time.time()
def log(m): print("[%5.0fs] %s" % (time.time() - T0, m), flush=True)

from sklearn.model_selection import GroupShuffleSplit
from ftfy import fix_text
KEY = ["entity", "chunk_text"]

def research_folds(df):
    """The research notebook's chunk-grouped split (its Cell 25): group = chunk_id if the file has one, else the (ftfy-fixed)
    chunk text (the entity when the chunk is empty); groups sorted, shuffled with SEED; first 15% = lockbox, the rest dealt
    round-robin into 5 CV folds. Returns one fold label per row of df (all rows, all coding systems)."""
    cid = next((c for c in df.columns if c.lower() == "chunk_id"), None)
    if cid is not None and (df[cid].astype(str) != "NA").any():
        grp = "CID::" + df[cid].astype(str)
    else:
        ch, en = df.chunk_text.map(fix_text), df.entity.map(fix_text)
        grp = pd.Series(np.where(ch.str.strip() != "", "TXT::" + ch, "ENT::" + en), index=df.index)
    groups = np.array(sorted(grp.unique()))
    groups = groups[np.random.default_rng(SEED).permutation(len(groups))]
    n_lock = int(round(0.15 * len(groups)))
    fold_of = {g: "lockbox" for g in groups[:n_lock]}
    fold_of.update({g: str(i % 5) for i, g in enumerate(groups[n_lock:])})
    return grp.map(fold_of)

def prepare(csv_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["fold"] = research_folds(df) if SPLIT == "lockbox" else ""
    keep = pd.concat([df.loc[df.subconcept == "DiagnosisCode", KEY + ["fold"]],
                      df.loc[df.coding_system == "ICD10CM", KEY + ["fold"]]]).drop_duplicates(KEY)
    icd = (df[df.coding_system == "ICD10CM"][KEY + ["code", "display"]]
           .rename(columns={"code": "icd10cm_code", "display": "icd10cm_display"}))
    sno = (df[df.coding_system == "SNOMEDCT_US"][KEY + ["code", "display"]]
           .rename(columns={"code": "snomed_code", "display": "snomed_display"}))
    wide = keep.merge(icd, on=KEY, how="left").merge(sno, on=KEY, how="left").fillna("")
    n_dup = int(wide.duplicated(KEY).sum())                 # v8: a mention with several gold codes would be duplicated by the merge
    if n_dup:
        print(f"WARNING: {n_dup} mentions have more than one gold code; keeping the first (evaluation counts one gold)")
        wide = wide.drop_duplicates(KEY).reset_index(drop=True)
    wide["row_id"] = range(len(wide))
    wide["grp"] = pd.factorize(wide.chunk_text)[0]         # v8: chunk group, used for chunk-grouped leave-out
    if SPLIT == "lockbox":
        got = wide[wide.icd10cm_code != ""].groupby("fold").size().to_dict()
        print("mentions with ICD gold per fold (this notebook | research notebook):",
              {f: (got.get(f, 0), e) for f, e in EXPECTED_ICD_PER_FOLD.items()})
        if any(abs(got.get(f, 0) - e) > 3 for f, e in EXPECTED_ICD_PER_FOLD.items()):
            print("WARNING: the split differs from the research notebook by more than a few mentions; "
                  "the two pipelines' lockbox numbers are then not on identical mentions")
        test = wide[wide.fold == "lockbox"]
        dev = wide[wide.fold == DEV_FOLD]
        train = wide[~wide.fold.isin(["lockbox", DEV_FOLD])]
    else:
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tv_i, te_i = next(g1.split(wide, groups=wide.chunk_text))
        tv, test = wide.iloc[tv_i], wide.iloc[te_i]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
        tr_i, dv_i = next(g2.split(tv, groups=tv.chunk_text))
        train, dev = tv.iloc[tr_i], tv.iloc[dv_i]
    train, dev, test = (x.reset_index(drop=True) for x in (train, dev, test))
    ov = [len(set(a.chunk_text) & set(b.chunk_text)) for a, b in ((train, dev), (train, test), (dev, test))]
    if any(ov):   # identical chunk text filed under different chunk_ids can land in two splits
        print(f"note: chunk texts shared between splits (train-dev, train-test, dev-test): {ov}")
    return wide, train, dev, test

wide, train, dev, test = prepare(CSV_PATH)
for name, part in (("train", train), ("dev", dev), ("test", test)):
    print(f"{name:5s} rows {len(part):5d} | ICD gold {(part.icd10cm_code != '').sum():5d} | SNOMED gold {(part.snomed_code != '').sum():5d}")


def load_dictionaries():
    p_icd, p_sn = f"{WORK_DIR}/icd.pkl", f"{WORK_DIR}/snomed.pkl"
    if os.path.exists(p_icd) and os.path.exists(p_sn):
        return pd.read_pickle(p_icd), pd.read_pickle(p_sn)
    log("reading the Excel dictionaries (first run only, several minutes)")
    icd_sheets = pd.read_excel(ICD_XLSX, sheet_name=None, dtype=str)
    sn_sheets = pd.read_excel(SNOMED_XLSX, sheet_name=None, dtype=str)
    icd = max(icd_sheets.values(), key=len).fillna("")
    sn = max(sn_sheets.values(), key=len).fillna("")
    assert {"Code", "Description"} <= set(icd.columns), icd.columns
    assert {"snomed_code", "description"} <= set(sn.columns), sn.columns
    icd.to_pickle(p_icd); sn.to_pickle(p_sn)
    return icd, sn

ICD, SN = load_dictionaries()
log(f"ICD rows {len(ICD):,} ({ICD.Code.nunique():,} codes) | SNOMED rows {len(SN):,} ({SN.snomed_code.nunique():,} concepts)")

ALLOWED_SNOMED_TAGS = {"disorder", "finding", "situation", "morphologic abnormality", "observable entity",
                       "procedure", "body structure", "qualifier value", "substance", "regime/therapy",
                       "organism", "event", "physical object"}

def snomed_tagmap(sn):
    """concept -> semantic tag, read from the trailing '(tag)' of its names; only tags that are common."""
    suf = sn.description.str.extract(r"\(([^()]+)\)\s*$")[0]
    cnt = sn.assign(suf=suf).dropna(subset=["suf"]).groupby("suf").snomed_code.nunique()
    known = set(cnt[cnt >= 150].index)
    f = sn.assign(suf=suf)
    f = f[f.suf.isin(known)].drop_duplicates("snomed_code", keep="last").set_index("snomed_code")
    return f.suf.to_dict()

TAGMAP = snomed_tagmap(SN)

# code -> readable text (used in prompts and error tables)
ICD_TEXT = ICD.drop_duplicates("Code").set_index("Code").Description.to_dict()
_o = pd.to_numeric(SN.get("description_order", pd.Series(1, index=SN.index)), errors="coerce").fillna(99)
SN_TEXT = (SN.assign(_o=_o).sort_values(["snomed_code", "_o"], kind="stable")
             .drop_duplicates("snomed_code").set_index("snomed_code").description.to_dict())

# ---- v5: ICD hierarchy from the code strings themselves (no external resource) ----
# 98% of gold ICD codes are LEAVES (billable). The dictionary also holds the parent/category codes (R05, I50, J96.0 ...)
# whose description is often the bare term ("Cough", "Heart failure"), so lexical search lands on the parent while the
# coder picked its unspecified child (R05.9, I50.9, J96.00). On training terms that exactly equal a dictionary
# description, the gold is reached 65% of the time from the raw match and 89% after descending to the default child.
_nz = lambda t: " ".join(re.sub(r"[^a-z0-9]+", " ", str(t).lower()).split())
_ICD_ALL = sorted(set(ICD.Code[~ICD.Code.str.contains("-")]))
_ICD_DESC = ICD.groupby("Code").Description.apply(list).to_dict()
ICD_LEAF = {c for i, c in enumerate(_ICD_ALL) if not (i + 1 < len(_ICD_ALL) and _ICD_ALL[i + 1].startswith(c))}
ICD_DEFAULT_CHILD, ICD_NLEAF = {}, {}
for i, p in enumerate(_ICD_ALL):
    if p in ICD_LEAF: continue
    kids, j = [], i + 1
    while j < len(_ICD_ALL) and _ICD_ALL[j].startswith(p):
        if _ICD_ALL[j] in ICD_LEAF: kids.append(_ICD_ALL[j])
        j += 1
    if not kids: continue
    un = [k for k in kids if any("unspecified" in str(x).lower() for x in _ICD_DESC.get(k, []))]
    nine = [k for k in kids if k.replace(".", "").endswith("9")]
    _ph = [_nz(x) for x in _ICD_DESC.get(p, []) if _nz(x)]
    _same_head = lambda k: any(_nz(x).startswith(h) for x in _ICD_DESC.get(k, []) for h in _ph)
    # v6: among the shortest 'unspecified' children prefer "<parent text>, unspecified" (I50 -> I50.9, not I50.1 'Left
    # ventricular failure, unspecified'); exact-description training hits reaching the gold: 88.8% -> 90.1%
    ICD_DEFAULT_CHILD[p] = min(un or nine or kids, key=lambda k: (len(k), 0 if _same_head(k) else 1, 0 if k.endswith("9") else 1, k))
    ICD_NLEAF[p] = len(kids)
log(f"ICD hierarchy: {len(ICD_LEAF):,} leaves, {len(ICD_DEFAULT_CHILD):,} parents with a default child "
    f"(e.g. R05 -> {ICD_DEFAULT_CHILD.get('R05')}, I50 -> {ICD_DEFAULT_CHILD.get('I50')}, J96.0 -> {ICD_DEFAULT_CHILD.get('J96.0')})")

# optional official SNOMED -> ICD-10-CM map (feature only, never the answer)
XMAP_FWD, XMAP_REV = {}, {}
if XMAP_PATH:
    _m = pd.read_csv(XMAP_PATH, sep="\t", dtype=str, keep_default_na=False)
    for s_, i_ in zip(_m["referencedComponentId"], _m["mapTarget"]):
        if i_:
            XMAP_FWD.setdefault(s_, set()).add(i_); XMAP_REV.setdefault(i_, set()).add(s_)
    log(f"official map loaded: {len(XMAP_FWD):,} SNOMED concepts")

# ---- v8: audit the gold codes against the dictionaries (teammate Cells 6-7) ----
from decimal import Decimal
_SCI = r"^\s*\d+(?:\.\d+)?[eE]\+?\d+\s*$"
_SN_IDS, _SN_TERMS = {}, {}
def _sn_lookup():
    """built only when a scientific-notation id is actually found"""
    if not _SN_IDS:
        u = pd.Series(SN.snomed_code.unique()); u = u[u.str.fullmatch(r"\d+")]
        _SN_IDS.update({L: np.sort(g.astype("int64").values) for L, g in u.groupby(u.str.len())})
        _SN_TERMS.update(SN.groupby("snomed_code").description.apply(lambda x: {str(t).lower() for t in x}).to_dict())

def repair_sci(sci, display):
    """'4.47661E+14' (Excel) keeps only 6 digits: find dictionary ids of the right length in the rounding window,
    then pick the one whose descriptions contain the gold display text."""
    _sn_lookup()
    mant, exp = sci.strip().upper().replace("+", "").split("E")
    exp, sig = int(exp), len(mant.replace(".", ""))
    val = Decimal(mant) * Decimal(10) ** exp; half = Decimal(10) ** (exp - sig + 1) / 2
    arr = _SN_IDS.get(exp + 1, np.array([], dtype="int64"))
    win = [str(x) for x in arr[np.searchsorted(arr, int(val - half)):np.searchsorted(arr, int(val + half), "right")]]
    hit = [c for c in win if str(display).strip().lower() in _SN_TERMS.get(c, set())]
    return hit[0] if len(hit) == 1 else (win[0] if len(win) == 1 else None)

def audit_and_repair(parts):
    for nm, part in parts.items():
        m = part.snomed_code.str.match(_SCI)
        if m.any():
            fixed = [repair_sci(c, d) for c, d in zip(part.snomed_code[m], part.snomed_display[m])]
            part.loc[m, "snomed_code"] = [f or c for f, c in zip(fixed, part.snomed_code[m])]
            print(f"{nm}: repaired {sum(f is not None for f in fixed)} of {int(m.sum())} scientific-notation SNOMED ids")
    allp = pd.concat(parts.values())
    for col, valid, name in (("icd10cm_code", set(ICD.Code), "ICD"), ("snomed_code", set(SN.snomed_code), "SNOMED")):
        g = allp[col][allp[col] != ""]
        bad = g[~g.isin(valid)]
        print(f"{name}: {len(g):,} gold codes, {100 * g.isin(valid).mean():.2f}% in the dictionary"
              + (f" | NOT in dictionary (can never be predicted): {bad.value_counts().head(10).to_dict()}" if len(bad) else ""))

audit_and_repair({"train": train, "dev": dev, "test": test})
VALID_CODES = {"icd": set(ICD.Code[~ICD.Code.str.contains("-")]), "snomed": set(SN.snomed_code)}


from rapidfuzz import process, fuzz

def norm(s):
    s = str(s).lower().replace("<br>", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())

def stem(t):
    if len(t) > 4 and t.endswith("ies"): return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ses"): return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")): return t[:-1]
    return t

def stem_text(raw): return " ".join(stem(t) for t in raw.split())
def nstem(s): return stem_text(norm(s))

# General medical abbreviations (public knowledge, NOT fitted to any evaluation data).
# Ambiguous ones (MS, RA, PID, PE ...) are deliberately left out: the chunk / LLM resolves them.
ABBREV = {
 "htn": "hypertension", "dm": "diabetes mellitus", "t2dm": "type 2 diabetes mellitus", "t1dm": "type 1 diabetes mellitus",
 "dm2": "type 2 diabetes mellitus", "dm1": "type 1 diabetes mellitus", "iddm": "insulin dependent diabetes mellitus",
 "niddm": "non insulin dependent diabetes mellitus", "dka": "diabetic ketoacidosis", "ckd": "chronic kidney disease",
 "aki": "acute kidney injury", "arf": "acute renal failure", "crf": "chronic renal failure", "esrd": "end stage renal disease",
 "copd": "chronic obstructive pulmonary disease", "chf": "congestive heart failure", "hf": "heart failure",
 "hfref": "heart failure with reduced ejection fraction", "hfpef": "heart failure with preserved ejection fraction",
 "cad": "coronary artery disease", "mi": "myocardial infarction", "stemi": "st elevation myocardial infarction",
 "nstemi": "non st elevation myocardial infarction", "afib": "atrial fibrillation", "svt": "supraventricular tachycardia",
 "dvt": "deep vein thrombosis", "gerd": "gastroesophageal reflux disease", "uti": "urinary tract infection",
 "uri": "upper respiratory infection", "urti": "upper respiratory tract infection", "lrti": "lower respiratory tract infection",
 "cva": "cerebrovascular accident", "tia": "transient ischemic attack", "ards": "acute respiratory distress syndrome",
 "osa": "obstructive sleep apnea", "bph": "benign prostatic hyperplasia", "hld": "hyperlipidemia", "hlp": "hyperlipidemia",
 "sob": "shortness of breath", "doe": "dyspnea on exertion", "ibs": "irritable bowel syndrome", "ibd": "inflammatory bowel disease",
 "pud": "peptic ulcer disease", "gib": "gastrointestinal bleeding", "ugib": "upper gastrointestinal bleeding",
 "sbo": "small bowel obstruction", "hcc": "hepatocellular carcinoma", "hiv": "human immunodeficiency virus", "tb": "tuberculosis",
 "pna": "pneumonia", "cap": "community acquired pneumonia", "hap": "hospital acquired pneumonia", "vap": "ventilator associated pneumonia",
 "ptsd": "post traumatic stress disorder", "adhd": "attention deficit hyperactivity disorder", "ocd": "obsessive compulsive disorder",
 "mdd": "major depressive disorder", "gad": "generalized anxiety disorder", "sle": "systemic lupus erythematosus",
 "oa": "osteoarthritis", "djd": "degenerative joint disease", "lbp": "low back pain", "lvh": "left ventricular hypertrophy",
 "pvd": "peripheral vascular disease", "pad": "peripheral arterial disease", "sirs": "systemic inflammatory response syndrome",
 "ich": "intracranial hemorrhage", "sah": "subarachnoid hemorrhage", "sdh": "subdural hematoma", "tbi": "traumatic brain injury",
 "als": "amyotrophic lateral sclerosis", "tach": "tachycardia", "brady": "bradycardia",
 "ams": "altered mental status", "loc": "loss of consciousness", "n v": "nausea vomiting", "rsv": "respiratory syncytial virus",
 "uc": "ulcerative colitis", "cld": "chronic liver disease", "nash": "nonalcoholic steatohepatitis", "pah": "pulmonary arterial hypertension",
}
# v6: "hcc" in these charts is the billing tag "(HCC)" (Hierarchical Condition Category), not hepatocellular carcinoma
ABBREV.pop("hcc", None)
# v6: more general, UNAMBIGUOUS clinical abbreviations (public knowledge; ambiguous ones such as ms, ra, pe, cp, lad, ttp, mr are left out)
ABBREV.update({
 # body regions / function (very frequent in these charts)
 "lle": "left lower extremity", "rle": "right lower extremity", "lue": "left upper extremity", "rue": "right upper extremity",
 "ble": "bilateral lower extremities", "bue": "bilateral upper extremities", "bilat": "bilateral", "abd": "abdominal",
 "adl": "activities of daily living", "adls": "activities of daily living", "iadl": "instrumental activities of daily living",
 "iadls": "instrumental activities of daily living", "rom": "range of motion", "arom": "active range of motion",
 "prom": "passive range of motion", "aarom": "active assisted range of motion", "nwb": "non weight bearing", "fx": "fracture",
 "hx": "history", "sx": "symptoms", "sz": "seizure", "ha": "headache",
 # cardiac
 "af": "atrial fibrillation", "afl": "atrial flutter", "aflutter": "atrial flutter", "vtach": "ventricular tachycardia",
 "vt": "ventricular tachycardia", "nsvt": "nonsustained ventricular tachycardia", "vfib": "ventricular fibrillation",
 "pvc": "premature ventricular contractions", "pvcs": "premature ventricular contractions", "pac": "premature atrial contractions",
 "pacs": "premature atrial contractions", "lbbb": "left bundle branch block", "rbbb": "right bundle branch block",
 "chb": "complete heart block", "sss": "sick sinus syndrome", "wpw": "wolff parkinson white syndrome",
 "ashd": "atherosclerotic heart disease", "ascvd": "atherosclerotic cardiovascular disease", "aaa": "abdominal aortic aneurysm",
 "asd": "atrial septal defect", "vsd": "ventricular septal defect", "pfo": "patent foramen ovale", "mvp": "mitral valve prolapse",
 "hcm": "hypertrophic cardiomyopathy", "icm": "ischemic cardiomyopathy", "nicm": "nonischemic cardiomyopathy",
 "cabg": "coronary artery bypass graft", "svg": "saphenous vein graft", "aicd": "automatic implantable cardioverter defibrillator",
 "ppm": "permanent pacemaker", "lvef": "left ventricular ejection fraction", "ef": "ejection fraction",
 "lae": "left atrial enlargement", "rvh": "right ventricular hypertrophy", "qtc": "prolonged qt interval",
 "vte": "venous thromboembolism", "phtn": "pulmonary hypertension", "pht": "pulmonary hypertension",
 "pnd": "paroxysmal nocturnal dyspnea",
 # respiratory / sleep
 "ild": "interstitial lung disease", "ipf": "idiopathic pulmonary fibrosis", "ohs": "obesity hypoventilation syndrome",
 "ahi": "apnea hypopnea index", "aom": "acute otitis media", "om": "otitis media",
 # renal / gu
 "atn": "acute tubular necrosis", "luts": "lower urinary tract symptoms", "oab": "overactive bladder", "pyelo": "pyelonephritis",
 # gi / liver
 "lgib": "lower gastrointestinal bleeding", "brbpr": "bright red blood per rectum", "brpr": "bright red blood per rectum",
 "nafld": "nonalcoholic fatty liver disease", "masld": "metabolic dysfunction associated steatotic liver disease",
 "hbv": "hepatitis b", "hcv": "hepatitis c", "cdiff": "clostridioides difficile infection",
 # infection
 "mrsa": "methicillin resistant staphylococcus aureus", "vre": "vancomycin resistant enterococcus",
 "esbl": "extended spectrum beta lactamase", "covid": "covid 19", "cmv": "cytomegalovirus", "ebv": "epstein barr virus",
 "hsv": "herpes simplex", "vzv": "varicella zoster",
 # heme / onc
 "ida": "iron deficiency anemia", "itp": "immune thrombocytopenia", "dic": "disseminated intravascular coagulation",
 "cll": "chronic lymphocytic leukemia", "cml": "chronic myeloid leukemia", "aml": "acute myeloid leukemia",
 "nhl": "non hodgkin lymphoma", "mgus": "monoclonal gammopathy of undetermined significance", "mds": "myelodysplastic syndrome",
 "mets": "metastases",
 # endocrine / metabolic
 "t2d": "type 2 diabetes", "dmt2": "type 2 diabetes mellitus", "dmii": "type 2 diabetes mellitus", "gdm": "gestational diabetes mellitus",
 "hypok": "hypokalemia", "hyperk": "hyperkalemia", "hypona": "hyponatremia", "hypomag": "hypomagnesemia", "chol": "cholesterol",
 # neuro / psych
 "nph": "normal pressure hydrocephalus", "mci": "mild cognitive impairment", "bppv": "benign paroxysmal positional vertigo",
 "edh": "epidural hematoma", "sud": "substance use disorder", "aud": "alcohol use disorder", "oud": "opioid use disorder",
 "etoh": "alcohol", "ftt": "failure to thrive",
 # musculoskeletal
 "ddd": "degenerative disc disease", "cts": "carpal tunnel syndrome", "pmr": "polymyalgia rheumatica", "gca": "giant cell arteritis",
 "tka": "total knee arthroplasty", "tkr": "total knee replacement", "tha": "total hip arthroplasty", "thr": "total hip replacement",
 # care status
 "dnr": "do not resuscitate", "dni": "do not intubate", "cgm": "continuous glucose monitor", "npo": "nothing by mouth",
 "cxr": "chest x ray",
})

# ---------------------------------------------------------------------------------------------------------------
# v7: LAY-TERM NORMALISATION. General patient / everyday language -> the clinical word a dictionary uses
# (public medical knowledge, written once; not fitted to dev or test rows). The clinical words are APPENDED.
LAY_PHRASES = [
 (r"heart attack", "myocardial infarction"), (r"mini stroke", "transient ischemic attack"),
 (r"\bstroke\b", "cerebrovascular accident cerebral infarction"),
 (r"high blood pressure", "hypertension"), (r"low blood pressure", "hypotension"),
 (r"high cholesterol", "hypercholesterolemia hyperlipidemia"), (r"high (blood )?sugars?", "hyperglycemia"),
 (r"low (blood )?sugars?", "hypoglycemia"), (r"sugar diabetes", "diabetes mellitus"),
 (r"short(ness)? of breath|trouble breathing|difficulty breathing|hard to breathe|can t breathe|breathless", "dyspnea"),
 (r"passed out|pass out|blacked out|black out|fainted|fainting|faint\b", "syncope"),
 (r"throwing up|threw up|throw up|puking|vomited", "vomiting"),
 (r"heart ?burn|acid reflux", "gastroesophageal reflux"), (r"blood clots?", "thrombosis embolism"),
 (r"clots? in (the |my )?lungs?", "pulmonary embolism"), (r"collapsed lung", "pneumothorax"),
 (r"kidney stones?", "calculus of kidney nephrolithiasis"), (r"gall ?stones?", "cholelithiasis"),
 (r"blood in (the |my )?urine", "hematuria"), (r"blood in (the |my )?stools?", "hematochezia blood in stool"),
 (r"pink eye", "conjunctivitis"), (r"sore throat", "pharyngitis"), (r"runny nose", "rhinorrhea"),
 (r"stuffy nose|blocked nose", "nasal congestion"), (r"ear infection", "otitis media"),
 (r"bladder infection", "cystitis urinary tract infection"), (r"yeast infection", "candidiasis"),
 (r"cold sores?", "herpes labialis"), (r"shingles", "herpes zoster"), (r"chicken ?pox", "varicella"),
 (r"\bflu\b", "influenza"), (r"(belly|stomach|tummy) ?(ache|pain)", "abdominal pain"),
 (r"joint pain", "arthralgia"), (r"muscle (pain|ache)s?", "myalgia"),
 (r"trouble sleeping|can t sleep|cannot sleep|difficulty sleeping", "insomnia"),
 (r"trouble swallowing|difficulty swallowing", "dysphagia"), (r"ringing in (the |my )?ears?", "tinnitus"),
 (r"(bed|pressure) ?sores?", "pressure ulcer"), (r"nose ?bleeds?", "epistaxis"),
 (r"enlarged heart", "cardiomegaly"), (r"irregular heart ?beat", "arrhythmia"),
 (r"(racing|fast) heart( rate| beat)?", "tachycardia"), (r"slow heart( rate| beat)?", "bradycardia"),
 (r"pulled muscle", "strain of muscle"), (r"memory loss", "memory impairment amnesia"),
 (r"bed ?wetting", "enuresis"), (r"(leaking|leaks|can t hold) (urine|pee)", "urinary incontinence"),
 (r"loose stools?", "diarrhea"), (r"(water|fluid) (in|on) (the |my )?lungs?", "pulmonary edema"),
 (r"thin bones|bone loss", "osteoporosis"), (r"wear and tear arthritis", "osteoarthritis"),
 (r"under ?active thyroid", "hypothyroidism"), (r"over ?active thyroid", "hyperthyroidism"),
 (r"low iron", "iron deficiency"), (r"hives", "urticaria"), (r"stomach ulcer", "gastric ulcer"),
 (r"slipped dis[ck]", "intervertebral disc displacement"), (r"pinched nerve", "radiculopathy nerve compression"),
 (r"frozen shoulder", "adhesive capsulitis of shoulder"), (r"tennis elbow", "lateral epicondylitis"),
 (r"flat feet|flat foot", "pes planus"), (r"bunions?", "hallux valgus"), (r"ingrown (toe)?nails?", "ingrowing nail"),
 (r"lazy eye", "amblyopia"), (r"crossed eyes", "strabismus"), (r"near ?sighted", "myopia"),
 (r"far ?sighted", "hypermetropia"), (r"double vision", "diplopia"), (r"hay fever", "allergic rhinitis"),
 (r"asthma attack", "asthma exacerbation"), (r"black eye", "contusion of eyelid and periocular area"),
 (r"\bfell\b|\bfall(ing)? down\b", "fall"),
]
LAY_PHRASES = [(re.compile(p), e) for p, e in LAY_PHRASES]
LAY_WORDS = {"belly": "abdominal", "tummy": "abdominal", "swollen": "swelling edema", "itchy": "pruritus",
             "itching": "pruritus", "itch": "pruritus", "bruise": "contusion", "bruised": "contusion",
             "bruising": "ecchymosis contusion", "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue",
             "dizzy": "dizziness", "lightheaded": "dizziness", "weak": "weakness", "sprained": "sprain",
             "constipated": "constipation", "anemic": "anemia", "depressed": "depression", "anxious": "anxiety",
             "fits": "seizure", "lump": "mass", "lumps": "mass", "tingling": "paresthesia", "pins": "paresthesia",
             "numb": "numbness", "obese": "obesity", "overweight": "overweight obesity", "feverish": "fever",
             "wheezy": "wheezing", "smoker": "tobacco use nicotine dependence", "alcoholic": "alcohol dependence",
             "drunk": "alcohol intoxication", "confused": "confusion", "forgetful": "memory impairment"}
# "broke / broken / cracked + body part" -> "fracture of <anatomical site>" (lay body word -> anatomical words)
BREAK_WORDS = {"broke", "broken", "break", "breaks", "cracked", "crack", "snapped", "fractured", "busted"}
BODY_ANAT = {"back": "spine vertebra", "spine": "spine vertebra", "neck": "cervical vertebra", "tailbone": "coccyx",
             "hip": "hip femur", "leg": "lower leg tibia", "shin": "tibia", "thigh": "femur", "arm": "arm humerus",
             "wrist": "wrist radius", "elbow": "elbow", "ankle": "ankle", "foot": "foot", "toe": "toe", "toes": "toe",
             "finger": "finger", "fingers": "finger", "hand": "hand", "collarbone": "clavicle", "collar": "clavicle",
             "rib": "rib", "ribs": "rib", "jaw": "mandible jaw", "nose": "nasal bone", "skull": "skull",
             "kneecap": "patella", "knee": "knee patella", "shoulder": "shoulder humerus", "pelvis": "pelvis",
             "heel": "calcaneus", "cheekbone": "zygoma", "tooth": "tooth", "teeth": "tooth"}

def lay_expand(raw):
    """raw = normalised term. Returns clinical words to APPEND (never replaces anything)."""
    out = [e for p, e in LAY_PHRASES if p.search(raw)]
    toks = raw.split()
    out += [LAY_WORDS[t] for t in toks if t in LAY_WORDS]
    if any(t in BREAK_WORDS for t in toks):
        sites = [BODY_ANAT[t] for t in toks if t in BODY_ANAT]
        out += [f"fracture of {s}" for s in sites] or ["fracture"]
    seen, res = set(), []
    for e in out:
        if e not in seen and e not in raw: seen.add(e); res.append(e)
    return res

# ---------------------------------------------------------------------------------------------------------------
# v7: BODY REGIONS (for the agreement features). Words are stemmed like the queries. A word may belong to 2 regions.
_REGIONS = {
 "head": "head skull scalp face facial cranial forehead temple",
 "brain": "brain cerebral cerebrum intracranial cerebellar cerebellum",
 "eye": "eye eyes ocular eyelid orbit orbital retina retinal cornea corneal conjunctiva conjunctival lens vision visual",
 "ear": "ear ears auricular otic tympanic cochlear hearing",
 "nose": "nose nasal sinus nostril",
 "mouth": "mouth oral tooth teeth dental gum gingival tongue lip jaw mandible maxilla",
 "throat": "throat pharynx pharyngeal larynx laryngeal tonsil vocal",
 "neck": "neck",
 "spine": "back spine spinal vertebra vertebral vertebrae lumbar sacral lumbosacral coccyx sacrum disc dorsalgia",
 "chest": "chest thorax rib ribs sternum",
 "breast": "breast mammary nipple",
 "heart": "heart cardiac coronary myocardial atrial ventricular mitral pericardial",
 "lung": "lung lungs pulmonary bronchial bronchus pleural pleura",
 "abdomen": "abdomen abdominal belly umbilical",
 "gi": "stomach gastric esophagus esophageal intestine intestinal bowel colon colonic rectal rectum anal duodenal",
 "liver": "liver hepatic gallbladder biliary bile",
 "urinary": "kidney kidneys renal ureter bladder urinary urethra",
 "genital": "penis testis testicular scrotal scrotum prostate uterus uterine ovary ovarian vagina vaginal vulva cervix",
 "hip": "pelvis pelvic hip hips femur",
 "arm": "arm arms shoulder elbow wrist hand hands finger fingers thumb forearm humerus radius ulna clavicle scapula",
 "leg": "leg legs knee knees ankle ankles foot feet toe toes thigh calf shin tibia fibula patella heel femur",
}
REGION_OF = {}
for _r, _ws in _REGIONS.items():
    for _w in _ws.split(): REGION_OF.setdefault(stem(_w), set()).add(_r)
def regions(tokens):
    out = set()
    for t in tokens:
        r = REGION_OF.get(t)
        if r: out |= r
    return frozenset(out)

# ---------------------------------------------------------------------------------------------------------------
# v7: NEGATION / HISTORY / FAMILY cues (NegEx-style: a few words before the term, inside the same sentence,
# or a negative word right after it). General clinical-NLP rules; nothing here was fitted to dev or test.
_NEG_PRE = re.compile(r"\b(no|not|denies|denied|deny|negative for|neg for|without|free of|absence of|absent|"
                      r"ruled out|never had|never)\b")
_HIST_PRE = re.compile(r"(?<!medical )(?<!surgical )\b(history of|hx of|h o|s p|status post|previous|prior|"
                       r"past history of|remote|healed)\b")
_FAM_ANY = re.compile(r"\b(family history|family hx|fhx|fh|famhx)\b")
_FAM_PRE = re.compile(r"\b(mother|father|mom|dad|brother|sister|sibling|siblings|son|daughter|grandmother|grandfather|"
                      r"maternal|paternal|aunt|uncle|cousin|parents?)\b")
_NEG_POST = re.compile(r"^\s*[:\-]?\s*(negative|neg|denied|denies|absent|none|not present|ruled out)\b")
_ENT_NEG = re.compile(r"^(no|not|without|absent|absence of|negative for|denies|free of)\b")
_ENT_HIST = re.compile(r"\b(history of|hx of|h o|s p|status post|personal history)\b")
_SENT = re.compile(r"[.;\n]|\bbut\b|\bhowever\b|\balthough\b")

def detect_cues(entity, chunk):
    """-> (negated, historical, family, cue_in_entity), each 0/1."""
    e = norm(entity)
    en, eh, ef = int(bool(_ENT_NEG.search(e))), int(bool(_ENT_HIST.search(e)) and not _FAM_ANY.search(e)), int(bool(_FAM_ANY.search(e)))
    if en or eh or ef: return en, eh, ef, 1
    c = str(chunk or "").replace("<br>", "\n").lower()
    i = c.find(str(entity).lower()) if entity else -1
    if i < 0: return 0, 0, 0, 0
    before = c[max(0, i - 200):i]
    clause = _SENT.split(before)[-1]                         # text since the last sentence break
    pre = norm(clause).split()[-6:]
    pre = " ".join(pre)
    big = norm(" ".join(re.split(r"[.;]", before)[-1:]))     # the whole sentence (family-history headers)
    after = c[i + len(entity): i + len(entity) + 30]
    neg = int(bool(_NEG_PRE.search(pre)) or bool(_NEG_POST.search(after)))
    fam = int(bool(_FAM_ANY.search(big)) or bool(_FAM_PRE.search(pre)))
    hist = int(bool(_HIST_PRE.search(pre)) and not fam)
    return neg, hist, fam, 0

def cand_cues(text):
    """candidate description (normalised, stemmed) -> (says absent, says history, says family history)"""
    fam = int("family history" in text or text.startswith("fh "))
    hist = int(not fam and ("history of" in text or "h o " in text or text.startswith("personal history") or "status post" in text))
    neg = int(text.startswith(("no ", "not ", "absence ", "absent ", "negative ", "without ", "free of ")))
    return neg, hist, fam

def cue_prefix(neg, hist, fam, in_entity):
    """words for the extra cue-aware search, only when the cue came from the CONTEXT (the term itself is unchanged)"""
    if in_entity: return ""
    return "family history of" if fam else "history of" if hist else "no" if neg else ""

def ctx_text(entity, chunk, width=None):
    """v7: the text around the term in its chunk (for the context channel); '' when there is no chunk"""
    width = CTX_CHARS if width is None else width
    c = str(chunk or "").replace("<br>", " ")
    if not c.strip(): return ""
    i = c.lower().find(str(entity).lower())
    s = c[:2 * width] if i < 0 else c[max(0, i - width): i + len(str(entity)) + width]
    return norm(s)

_PAREN = re.compile(r"([A-Za-z][A-Za-z\- ]{3,80}?)\s*\(([A-Za-z]{2,8})\)")
def chunk_abbrevs(chunk):
    """Definitions written in the chunk itself, e.g. 'chronic kidney disease (CKD)'. Initials must match."""
    out = {}
    for m in _PAREN.finditer(chunk or ""):
        words = m.group(1).lower().replace("-", " ").split()
        ab = m.group(2).lower(); n = len(ab)
        if len(words) >= n and "".join(w[0] for w in words[-n:]) == ab:
            out[ab] = " ".join(words[-n:])
    return out

HCC_TAG = re.compile(r"\(\s*(?:hhs-|cms-)?hcc\s*\)", re.I)       # v6: "(HCC)" billing tag

# v8 (teammate Cell 12): measurements and grades are not part of the diagnosis ("2.9 cm lesion", "+2 edema", "4/5 strength")
_NUM = r"\d+(?:\.\d+)?"
QTY_PATTERNS = [re.compile(p) for p in (
    rf"\b{_NUM}(?:\s*-\s*{_NUM})?\s*(?:x\s*{_NUM}\s*)?(?:cm|mm|ml|mg|kg|g|m)\b",     # 2.9 cm, 4-5 mm, 6.5 x 35 mm
    rf"\b{_NUM}(?:\s*-\s*{_NUM})?\s*(?:%|percent\b)",                                # 100%, 30%
    r"\b\d\s*[-+]?\s*(?:/|out of)\s*5\b",                                           # 4/5, 4-/5, 4+ out of 5
    r"(?<![\w])\d(?:\s*-\s*\d)?\s*\+(?![\w])",                                      # 1+, 1-2+
    r"(?<![\w])\+\s*\d\b")]                                                          # +2
def strip_qty(s):
    for p in QTY_PATTERNS: s = p.sub(" ", s)
    return s
_LETTER_DIGIT = re.compile(r"^([a-z]+)(\d+[a-z]?)$")                # v6: "ckd3", "ckd3b"

def _plausible_fix(w, r):
    """v6: accept a spelling correction only if it looks like a real typo of the same word."""
    if " " in r: return True                                         # glued words: "chronickidney" -> "chronic kidney"
    if w[:2] != r[:2] or abs(len(w) - len(r)) > 2: return False      # blocks rashes->ashes, aflutter->flutter
    for p in ("non", "anti", "un"):                                   # blocks nonintractable->intractable
        if w.startswith(p) != r.startswith(p): return False
    return True

class TextPrep:
    """entity (+chunk) -> (raw query, stemmed query). Expansions AND spelling corrections are APPENDED (v6):
    the original words are never replaced."""
    def __init__(self, vocab_words):
        self.vocab = set(vocab_words); self.vlist = sorted(self.vocab); self.cache = {}
    def fix_token(self, w):
        if w in self.cache: return self.cache[w]
        r = w
        if w.isalpha() and len(w) >= 4 and w not in self.vocab and w not in ABBREV:
            split = None
            for i in range(4, len(w) - 3):                      # glued words: "chronickidney"
                if w[:i] in self.vocab and w[i:] in self.vocab: split = w[:i] + " " + w[i:]; break
            if split: r = split
            else:                                               # typo: nearest dictionary word
                m = process.extractOne(w, self.vlist, scorer=fuzz.ratio, score_cutoff=88)
                if m and _plausible_fix(w, m[0]): r = m[0]
        self.cache[w] = r
        return r
    def __call__(self, entity, chunk=""):
        raw = norm(HCC_TAG.sub(" ", str(entity)))
        if USE_QTY_STRIP:                                            # v8: drop measurements / grades (keep the term if nothing is left)
            raw = norm(strip_qty(HCC_TAG.sub(" ", str(entity)).lower())) or raw
        toks = raw.split()
        add = []
        if USE_SPELL:                                                # v6: append, never replace
            fixes = [self.fix_token(t) for t in toks]
            add += [f for t, f in zip(toks, fixes) if f != t]
        if USE_ABBREV:
            local = chunk_abbrevs(chunk)
            for t in toks:
                e = local.get(t) or ABBREV.get(t)
                if not e:
                    m = _LETTER_DIGIT.match(t)
                    if m and m.group(1) in ABBREV: e = ABBREV[m.group(1)] + " " + m.group(2)
                if e: add.append(e)
        if USE_LAY:                                                  # v7: lay language -> clinical words (appended)
            add += lay_expand(raw)
        q = (raw + " " + " ".join(add)).strip()
        return q, stem_text(q)

# ---- dictionary side: normalised texts per coding system -----------------
def build_dict_arrays():
    out = {}
    d = ICD[~ICD.Code.str.contains("-")]
    raws = [norm(x) for x in d.Description.values]
    out["icd"] = (d.Code.values, raws, [stem_text(r) for r in raws])
    s = SN.assign(tag=SN.snomed_code.map(TAGMAP))
    s = s[s.tag.isin(ALLOWED_SNOMED_TAGS)]
    base = [re.sub(r"\s*\([^()]+\)\s*$", "", x) if x.rstrip().endswith(f"({t})") else x
            for x, t in zip(s.description.values, s.tag.values)]
    raws = [norm(x) for x in base]
    out["snomed"] = (s.snomed_code.values, raws, [stem_text(r) for r in raws])
    return out

DICT_ARRAYS = build_dict_arrays()
_cnt = collections.Counter(t for sys_ in DICT_ARRAYS.values() for r in sys_[1] for t in r.split())
VOCAB = {w for w, c in _cnt.items() if c >= 2 and w.isalpha() and len(w) >= 3}
_tr_tok = collections.Counter(t for e in train.entity for t in norm(e).split())    # v6: words coders really use are not "typos"
VOCAB |= {w for w, c in _tr_tok.items() if c >= 2 and w.isalpha() and len(w) >= 3}
PREP = TextPrep(VOCAB)
log(f"dictionary texts ready: ICD {len(DICT_ARRAYS['icd'][0]):,} | SNOMED {len(DICT_ARRAYS['snomed'][0]):,} | vocab {len(VOCAB):,}")

def add_query_columns(df):
    q = [PREP(e, c) for e, c in zip(df.entity.values, df.chunk_text.values)]
    df = df.copy(); df["q_raw"] = [a for a, b in q]; df["q_stem"] = [b for a, b in q]
    cu = np.array([detect_cues(e, c) for e, c in zip(df.entity.values, df.chunk_text.values)], dtype=int).reshape(-1, 4)
    df["cue_neg"], df["cue_hist"], df["cue_fam"], df["cue_in_entity"] = cu[:, 0], cu[:, 1], cu[:, 2], cu[:, 3]
    return df

train, dev, test = (add_query_columns(x) for x in (train, dev, test))
print(train[["entity", "q_raw"]].sample(8, random_state=1).to_string())
for _n, _p in (("train", train), ("dev", dev)):
    print(f"v7 cues in {_n}: negated {_p.cue_neg.mean():.1%} | history {_p.cue_hist.mean():.1%} | family {_p.cue_fam.mean():.1%} "
          f"| lay words found in {np.mean([bool(lay_expand(norm(e))) for e in _p.entity]):.1%} of terms")
print("v7 check:", [PREP(e)[0] for e in ["broke my back", "heart attack", "passed out at home", "swollen ankles"]])
print("v7 cues :", [detect_cues(e, c) for e, c in [("diabetes", "Family history: mother with diabetes."),
      ("chest pain", "Patient denies chest pain or SOB."), ("stroke", "History of stroke in 2019."), ("htn", "PMH: HTN")]])
print("v8 check:", [PREP(e)[0] for e in ["+2 bilat le edema", "2.9 cm hypoattenuating lesion", "4/5 strength", "100% occlusion"]])
print("v6 check:", [PREP(e)[0] for e in ["acute respiratory failure (hcc)", "nonintractable migraine", "skin rashes", "ckd3", "lle edema"]])

from sklearn.feature_extraction.text import TfidfVectorizer

class Index:
    """Dictionary descriptions + 'train memory' rows (training entities act as extra synonyms of their gold code).
    Every score is the BEST over a code's descriptions, so a code with many synonyms is not penalised."""

    def __init__(self, dict_codes, dict_texts, dict_raw, syn_codes, syn_texts, syn_raw, syn_rowids, encoder=None, enc_b=None,
                 syn_groups=None):
        O = lambda a: np.asarray(a, dtype=object)
        d = pd.DataFrame({"code": O(dict_codes), "text": O(dict_texts), "raw": O(dict_raw)}).drop_duplicates(["code", "text"])
        d = d[d.text.str.len() > 0].sort_values("code", kind="stable").reset_index(drop=True)
        syn_codes = O(syn_codes)
        # v8: only dictionary codes are answerable (a predicted code outside the dictionary is scored wrong), so training
        # rows whose gold is not in the dictionary are left out of the train memory instead of being injected as new codes
        self.d = d
        self.codes, self.starts = np.unique(d.code.values, return_index=True)
        self.code_pos = {c: i for i, c in enumerate(self.codes)}

        self.wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, dtype=np.float32)
        self.Mw = self.wv.fit_transform(d.text.values).tocsr()
        self.Mc = self.cv.fit_transform(d.text.values).tocsr()

        grp = np.asarray(syn_groups) if syn_groups is not None else -1 - np.arange(len(syn_codes))
        s = pd.DataFrame({"code": syn_codes, "text": O(syn_texts), "raw": O(syn_raw), "rowid": np.asarray(syn_rowids), "grp": grp})
        s = s[(s.text.str.len() > 0) & s.code.isin(set(self.code_pos))].sort_values("code", kind="stable").reset_index(drop=True)
        self.syn = s
        self.syn_codes, self.syn_starts = np.unique(s.code.values, return_index=True)
        self.syn_code_idx = np.array([self.code_pos[c] for c in self.syn_codes])
        self.Sw = self.wv.transform(s.text.values).tocsr()
        self.Sc = self.cv.transform(s.text.values).tocsr()
        self.syn_rowid = s.rowid.values
        self.syn_grp = s.grp.values

        self.encoder = encoder
        if encoder is not None:                               # dense channel works on un-stemmed text
            self.Md = encoder(list(self.d.raw.values))
            self.Sd = encoder(list(s.raw.values))
        self.enc_b = enc_b
        if enc_b is not None:                                 # second dense channel (BioLORD), also un-stemmed text
            self.Mb = enc_b(list(self.d.raw.values))
            self.Sb = enc_b(list(s.raw.values))
        self._to_gpu()

    def _to_gpu(self):
        """Keep the two big dictionary matrices on the GPU (half precision) and free their RAM copies."""
        try:
            import torch
            if not torch.cuda.is_available(): return
            for name in ("Md", "Mb"):
                M = getattr(self, name, None)
                if isinstance(M, np.ndarray):
                    setattr(self, name, torch.from_numpy(M).to("cuda").half())
            gc.collect()
        except Exception as e:
            log(f"GPU matrices unavailable ({e!r}); keeping them in RAM")

    @staticmethod
    def _mm(M, Q):
        """dictionary matrix (RAM or GPU) x query vectors -> numpy (rows, n_queries)"""
        if isinstance(M, np.ndarray): return M @ Q.T
        import torch
        q = torch.from_numpy(np.ascontiguousarray(Q)).to(M.device).half()
        return (M @ q.T).float().cpu().numpy()

    def _loo_masks(self, rids, grps):
        """v8: train-memory rows hidden from each query: the row itself and (GROUP_LOO) every row of the same chunk."""
        out = []
        for j, rid in enumerate(rids):
            m = self.syn_rowid == rid
            if grps is not None and GROUP_LOO: m = m | (self.syn_grp == grps[j])
            out.append(m)
        return out

    def retrieve(self, texts, raw_texts=None, query_rowids=None, K=50, batch=16, qvec=None, k_extra=None,
                 ctx_raws=None, cue_stems=None, cue_raws=None, query_groups=None):
        """Per query: a DataFrame of candidate codes with lexical (dict_*, syn_*), SapBERT (dict_d, syn_d), BioLORD
        (dict_b, syn_b) and tuned-SapBERT (dict_q) scores. qvec = tuned query vectors (one row per query).
        query_rowids -> leave-one-out: a training row never finds itself in the train memory."""
        k_extra = K_EXTRA if k_extra is None else k_extra
        out, U = [], len(self.codes)
        for b0 in range(0, len(texts), batch):
            tb = texts[b0:b0 + batch]
            Qw, Qc = self.wv.transform(tb), self.cv.transform(tb)
            dw = np.maximum.reduceat((self.Mw @ Qw.T).toarray(), self.starts, axis=0)
            dc = np.maximum.reduceat((self.Mc @ Qc.T).toarray(), self.starts, axis=0)
            sw_rows = (self.Sw @ Qw.T).toarray(); sc_rows = (self.Sc @ Qc.T).toarray()
            rids = query_rowids[b0:b0 + batch] if query_rowids is not None else None
            masks = None
            if rids is not None:
                masks = self._loo_masks(rids, query_groups[b0:b0 + batch] if query_groups is not None else None)
                for j, m in enumerate(masks):
                    sw_rows[m, j] = 0.0; sc_rows[m, j] = 0.0
            sw = np.zeros((U, len(tb)), dtype=np.float32); sc = np.zeros((U, len(tb)), dtype=np.float32)
            sw[self.syn_code_idx] = np.maximum.reduceat(sw_rows, self.syn_starts, axis=0)
            sc[self.syn_code_idx] = np.maximum.reduceat(sc_rows, self.syn_starts, axis=0)
            comb = np.maximum(0.5 * dw + 0.5 * dc, 0.5 * sw + 0.5 * sc)
            if self.encoder is not None:
                Qd = self.encoder(list(raw_texts[b0:b0 + batch]))
                dd = np.maximum.reduceat(self._mm(self.Md, Qd), self.starts, axis=0)
                sd_rows = self.Sd @ Qd.T
                if masks is not None:
                    for j, m in enumerate(masks):
                        sd_rows[m, j] = 0.0
                sd = np.zeros((U, len(tb)), dtype=np.float32)
                sd[self.syn_code_idx] = np.maximum.reduceat(sd_rows, self.syn_starts, axis=0)
                dcomb = np.maximum(dd, sd)
            if self.enc_b is not None:
                Qb = self.enc_b(list(raw_texts[b0:b0 + batch]))
                db = np.maximum.reduceat(self._mm(self.Mb, Qb), self.starts, axis=0)
                sb_rows = self.Sb @ Qb.T
                if masks is not None:
                    for j, m in enumerate(masks):
                        sb_rows[m, j] = 0.0
                sb = np.zeros((U, len(tb)), dtype=np.float32)
                sb[self.syn_code_idx] = np.maximum.reduceat(sb_rows, self.syn_starts, axis=0)
                bcomb = np.maximum(db, sb)
            if qvec is not None:
                dq = np.maximum.reduceat(self._mm(self.Md, qvec[b0:b0 + batch]), self.starts, axis=0)
            dx = cu = None
            if ctx_raws is not None and self.encoder is not None:          # v7: context channel (dictionary only -> no leak)
                cb = list(ctx_raws[b0:b0 + batch]); mk = np.array([bool(t) for t in cb], np.float32)
                if mk.any():
                    Qx = self.encoder([t or "none" for t in cb])
                    dx = np.maximum.reduceat(self._mm(self.Md, Qx), self.starts, axis=0) * mk
            if cue_stems is not None:                                     # v7: cue-aware search ("history of <term>")
                cs_, cr_ = list(cue_stems[b0:b0 + batch]), list(cue_raws[b0:b0 + batch])
                mk = np.array([bool(t) for t in cs_], np.float32)
                if mk.any():
                    Qw2, Qc2 = self.wv.transform(cs_), self.cv.transform(cs_)
                    cu = 0.5 * np.maximum.reduceat((self.Mw @ Qw2.T).toarray(), self.starts, axis=0) \
                       + 0.5 * np.maximum.reduceat((self.Mc @ Qc2.T).toarray(), self.starts, axis=0)
                    if self.encoder is not None:
                        Qe = self.encoder([t or "none" for t in cr_])
                        cu = np.maximum(cu, np.maximum.reduceat(self._mm(self.Md, Qe), self.starts, axis=0))
                    cu = cu * mk
            for j in range(len(tb)):
                col = comb[:, j]
                top = np.argpartition(-col, K)[:K]
                if self.encoder is not None:                  # union of all channels' shortlists
                    top = np.union1d(top, np.argpartition(-dcomb[:, j], K)[:K])
                if self.enc_b is not None:
                    top = np.union1d(top, np.argpartition(-bcomb[:, j], k_extra)[:k_extra])
                if qvec is not None:
                    top = np.union1d(top, np.argpartition(-dq[:, j], k_extra)[:k_extra])
                if dx is not None and dx[:, j].any():
                    base_d = dcomb[:, j] if self.encoder is not None else col
                    top = np.union1d(top, np.argpartition(-(base_d + CTX_W * dx[:, j]), K_CTX)[:K_CTX])
                if cu is not None and cu[:, j].any():
                    top = np.union1d(top, np.argpartition(-cu[:, j], K_CUE)[:K_CUE])
                cand = top[np.argsort(-col[top], kind="stable")]
                f = {"code": self.codes[cand], "score": col[cand],
                     "dict_w": dw[cand, j], "dict_c": dc[cand, j], "syn_w": sw[cand, j], "syn_c": sc[cand, j],
                     "in_top": np.ones(len(cand), dtype=bool)}
                if self.encoder is not None:
                    f.update({"dict_d": dd[cand, j], "syn_d": sd[cand, j], "score_d": dcomb[cand, j]})
                if self.enc_b is not None:
                    f.update({"dict_b": db[cand, j], "syn_b": sb[cand, j], "score_b": bcomb[cand, j]})
                if qvec is not None:
                    f["dict_q"] = dq[cand, j]
                if ctx_raws is not None and self.encoder is not None:
                    f["ctx_d"] = dx[cand, j] if dx is not None else np.zeros(len(cand), np.float32)
                if cue_stems is not None:
                    f["cue_s"] = cu[cand, j] if cu is not None else np.zeros(len(cand), np.float32)
                out.append(pd.DataFrame(f))
        return out

class SapEncoder:
    """SapBERT [CLS] embeddings, L2-normalised. Big batches are cached on disk by content hash."""
    def __init__(self, name=SAPBERT_NAME, bs=512, max_len=32):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch, self.bs, self.max_len = torch, bs, max_len
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).to(self.dev).eval()
        if self.dev == "cuda": self.m.half()
        log(f"SapBERT loaded on {self.dev}")
    def _encode(self, texts):
        torch = self.torch
        order = np.argsort([len(t) for t in texts], kind="stable")
        chunks = []
        for i in range(0, len(texts), self.bs):
            idx = order[i:i + self.bs]
            t = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**t).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(h.float(), dim=1).cpu().numpy())
        E = np.concatenate(chunks); R = np.empty_like(E); R[order] = E
        return R.astype(np.float32)
    def __call__(self, texts):
        texts = [t if t else " " for t in texts]
        if len(texts) < 2000: return self._encode(texts)
        key = hashlib.md5("\n".join(texts).encode()).hexdigest()
        path = f"{EMB_DIR}/emb_{key}.npy"
        if os.path.exists(path): return np.load(path)
        log(f"encoding {len(texts):,} texts with SapBERT")
        E = self._encode(texts); np.save(path, E); return E

ENCODER = SapEncoder() if USE_DENSE else None


class MeanEncoder(SapEncoder):
    """Sentence-transformer style encoder (mean pooling), used for BioLORD-2023. Cached on disk like SapEncoder."""
    def __init__(self, name, bs=512, max_len=48, tag="bl"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch, self.bs, self.max_len, self.tag = torch, bs, max_len, tag
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).to(self.dev).eval()
        if self.dev == "cuda": self.m.half()
        log(f"{name} loaded on {self.dev}")
    def _encode(self, texts):
        torch = self.torch
        order = np.argsort([len(t) for t in texts], kind="stable")
        chunks = []
        for i in range(0, len(texts), self.bs):
            idx = order[i:i + self.bs]
            t = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**t).last_hidden_state
                msk = t["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
            chunks.append(torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy())
        E = np.concatenate(chunks); R = np.empty_like(E); R[order] = E
        return R.astype(np.float32)
    def __call__(self, texts):
        texts = [t if t else " " for t in texts]
        if len(texts) < 2000: return self._encode(texts)
        key = hashlib.md5(("\n".join(texts) + self.tag).encode()).hexdigest()
        path = f"{EMB_DIR}/{self.tag}_{key}.npy"
        if os.path.exists(path): return np.load(path)
        log(f"encoding {len(texts):,} texts with {self.tag}")
        E = self._encode(texts); np.save(path, E); return E

ENCODER_B = None
if USE_DENSE and USE_BIOLORD:
    try:
        ENCODER_B = MeanEncoder(BIOLORD_NAME)
    except Exception as e:
        log(f"BioLORD could not be loaded ({e!r}); continuing without it")

import lightgbm as lgb

LATERAL = {"left": "L", "right": "R", "bilateral": "B", "bilat": "B"}
UNSPEC = {"unspecified", "nos", "unsp"}
def lat_set(tokens): return {LATERAL[t] for t in tokens if t in LATERAL}

def context_window(entity, chunk, width=12):
    """Words within +-12 tokens of the entity in its chunk (the evidence for specificity decisions)."""
    ne, nc = nstem(entity), nstem(chunk)
    if not nc: return set()
    pos = nc.find(ne) if ne else -1
    toks = nc.split()
    if pos < 0: return set(toks[:40])
    return set(nc[:pos].split()[-width:]) | set(nc[pos + len(ne):].split()[:width])

class Featurizer:
    def __init__(self, index, system, train_df, code_col, tagmap=None, min_text_score=0.12):
        self.ix, self.system, self.tagmap, self.min_text_score = index, system, tagmap, min_text_score
        tr = train_df[train_df[code_col] != ""]
        self.code_count = tr[code_col].value_counts().to_dict()
        self.cat = (lambda c: c.split(".")[0]) if system == "icd" else (lambda c: c)
        self.cat_count = tr[code_col].map(self.cat).value_counts().to_dict() if system == "icd" else {}
        g_ = tr.grp.values if "grp" in tr.columns else -1 - np.arange(len(tr))      # v8: counts per (chunk, code / category)
        self.grp_code = collections.Counter(zip(g_, tr[code_col].values))
        self.grp_cat = collections.Counter(zip(g_, tr[code_col].map(self.cat).values)) if system == "icd" else {}
        self.tok_cache = {}
        ends = np.append(index.starts[1:], len(index.d))
        self.span = {c: (s, e) for c, s, e in zip(index.codes, index.starts, ends)}
        self.tags = sorted(ALLOWED_SNOMED_TAGS); self.tag_id = {t: i for i, t in enumerate(self.tags)}
    def descs(self, code):
        if code not in self.tok_cache:
            s, e = self.span[code]
            self.tok_cache[code] = [(t, frozenset(t.split())) for t in self.ix.d.text.values[s:e]]
        return self.tok_cache[code]

    def build(self, qdf, entity_col, chunk_col, gold_col, res, loo=False, orig_col=None):
        """entity_col holds the prepared (stemmed) query; orig_col the raw entity (for locating it in the chunk)."""
        rows = []
        origs = qdf[orig_col or entity_col].values
        grps = qdf["grp"].values if "grp" in qdf.columns else None
        cues = np.stack([qdf[c].values.astype(int) if c in qdf.columns else np.zeros(len(qdf), int)
                         for c in ("cue_neg", "cue_hist", "cue_fam", "cue_in_entity")], axis=1)
        for qi, (ent, orig, chunk, gold) in enumerate(zip(qdf[entity_col].values, origs, qdf[chunk_col].values, qdf[gold_col].values)):
            E_text = ent; E = frozenset(E_text.split()); Elat = lat_set(E); Ereg = regions(E)
            qn, qh, qf, qe = (int(x) for x in cues[qi])
            W = context_window(orig, chunk)
            cand = res[qi]
            top = float(cand.score.iloc[0]) if len(cand) else 0.0
            has_d = "score_d" in cand.columns
            top_d = float(cand.score_d.max()) if has_d else 0.0
            has_b = "score_b" in cand.columns
            top_b = float(cand.score_b.max()) if has_b else 0.0
            has_q = "dict_q" in cand.columns
            top_q = float(cand.dict_q.max()) if has_q else 0.0
            has_x = "ctx_d" in cand.columns
            top_x = float(cand.ctx_d.max()) if has_x and len(cand) else 0.0
            has_cu = "cue_s" in cand.columns
            cat_mass = {}
            for c, s in zip(cand.code.values, cand.score.values):
                cat_mass[self.cat(c)] = cat_mass.get(self.cat(c), 0.0) + float(s)
            for rank, r in enumerate(cand.itertuples(index=False)):
                code = r.code
                cnt = self.code_count.get(code, 0)
                catc = self.cat_count.get(self.cat(code), 0) if self.system == "icd" else 0
                if loo and GROUP_LOO and grps is not None:  # v8: leave the whole chunk out of the priors
                    cnt -= self.grp_code.get((grps[qi], code), 0)
                    if self.system == "icd": catc -= self.grp_cat.get((grps[qi], self.cat(code)), 0)
                elif loo and code == gold:                 # leave-one-out priors for training rows
                    cnt -= 1
                    if self.system == "icd": catc -= 1
                f = {"qid": qi, "code": code, "label": int(code == gold),
                     "score": r.score, "score_gap": top - r.score, "rank": rank,
                     "dict_w": r.dict_w, "dict_c": r.dict_c, "syn_w": r.syn_w, "syn_c": r.syn_c,
                     "code_prior": math.log1p(max(cnt, 0)), "cat_prior": math.log1p(max(catc, 0)),
                     "cat_mass": cat_mass.get(self.cat(code), 0.0) / max(top, 1e-6),
                     "n_desc": self.span[code][1] - self.span[code][0],
                     "e_len": len(E), "e_chars": len(E_text), "e_digit": int(any(ch.isdigit() for ch in E_text)),
                     "code_len": len(code), "code_unspec": int(any(T & UNSPEC for _, T in self.descs(code)))}
                if self.system == "snomed":
                    f["tag"] = self.tag_id.get(self.tagmap.get(code, ""), -1)
                else:                                     # v5: hierarchy / specificity features
                    f["is_leaf"] = int(code in ICD_LEAF or code not in ICD_DEFAULT_CHILD)
                    f["n_leaf_below"] = ICD_NLEAF.get(code, 0)
                    f["inh"] = int(getattr(r, "inh", 0))
                    f["dcp"] = float(getattr(r, "dcp", 0.0))
                if has_d:
                    f["dict_d"], f["syn_d"], f["score_d"] = r.dict_d, r.syn_d, r.score_d
                    f["score_d_gap"] = top_d - r.score_d
                if has_b:
                    f["dict_b"], f["syn_b"], f["score_b"] = r.dict_b, r.syn_b, r.score_b
                    f["score_b_gap"] = top_b - r.score_b
                if has_q:
                    f["dict_q"] = r.dict_q
                    f["dq_gap"] = top_q - r.dict_q
                if has_x:                                  # v7: chunk-context similarity
                    f["ctx_d"] = r.ctx_d; f["ctx_d_gap"] = top_x - r.ctx_d
                if has_cu:
                    f["cue_s"] = r.cue_s
                if True:                      # text-overlap features for every candidate
                    best = None
                    for t, T in self.descs(code):
                        inter = len(E & T); jac = inter / max(1, len(E | T)); key = (jac, -len(T))
                        if best is None or key > best[0]: best = (key, t, T, inter)
                    (jac, _), t, T, inter = best
                    extra = T - E; Tlat = lat_set(T)
                    f.update({"jac": jac, "exact": int(any(t2 == E_text for t2, _ in self.descs(code))),
                              "cov_e": inter / max(1, len(E)), "cov_d": inter / max(1, len(T)),
                              "n_extra": len(extra), "n_missing": len(E - T),
                              "ctx_hit": len(extra & W), "ctx_frac": len(extra & W) / max(1, len(extra)),
                              "lat_match": int(bool(Elat) and Elat == Tlat),
                              "lat_conflict": int(bool(Elat) and bool(Tlat) and Elat != Tlat and "B" not in Elat | Tlat),
                              "lat_extra": int(not Elat and bool(Tlat)), "unspec": int(bool(T & UNSPEC))})
                    if USE_BODY:                          # v7: does the candidate talk about the same body part?
                        Treg = regions(T)
                        f.update({"body_match": int(bool(Ereg & Treg)),
                                  "body_conflict": int(bool(Ereg) and bool(Treg) and not (Ereg & Treg)),
                                  "body_extra": int(not Ereg and bool(Treg)), "body_missing": int(bool(Ereg) and not Treg)})
                    if USE_CUES:                          # v7: negation / history / family agreement
                        cn, ch, cf = cand_cues(t)
                        f.update({"q_neg": qn, "q_hist": qh, "q_fam": qf, "q_cue_ent": qe, "c_neg": cn, "c_hist": ch,
                                  "c_fam": cf, "neg_agree": int(qn == cn), "hist_agree": int(qh == ch), "fam_agree": int(qf == cf)})
                rows.append(f)
        return pd.DataFrame(rows)

_DEFAULT_OF = {}
def _is_default_child(code):
    """1 if the code is the default ('unspecified') child of one of its ICD ancestors (R05.9 of R05, J96.00 of J96.0)."""
    if code not in _DEFAULT_OF:
        a, v = code, 0
        while len(a) > 3:
            a = a[:-1].rstrip(".")
            if ICD_DEFAULT_CHILD.get(a) == code: v = 1; break
        _DEFAULT_OF[code] = v
    return _DEFAULT_OF[code]

def add_sibling_features(F, system):
    """v6: compare each candidate with the OTHER candidates of the same term and, for ICD, with its siblings in the same
    3-character category - the 'which child of this category' decision (10 ICD points are lost there on dev)."""
    F = F.copy()
    for col in ("jac", "cov_e", "score", "ctx_d", "cue_s"):
        if col in F.columns: F[col + "_rel_q"] = F[col] - F.groupby("qid")[col].transform("max")
    if "n_extra" in F.columns:
        F["n_extra_rel_q"] = F.n_extra - F.groupby("qid").n_extra.transform("min")
    if system == "icd":
        cat = F.code.str[:3]
        g = F.groupby([F.qid, cat])
        F["sib_n"] = g.code.transform("size")                       # how many siblings compete
        F["sib_rank"] = g.score.rank(ascending=False, method="first")
        F["sib_gap"] = g.score.transform("max") - F.score
        F["sib_extra_rel"] = F.n_extra - g.n_extra.transform("min")  # more specific than its least-specific sibling?
        F["sib_jac_rel"] = F.jac - g.jac.transform("max")
        F["sib_has_unspec"] = g.unspec.transform("max")             # is there an 'unspecified' sibling to fall back to?
        F["is_default_child"] = F.code.map(_is_default_child).astype(int)
        F["p_code_in_cat"] = np.expm1(F.code_prior) / np.maximum(1.0, np.expm1(F.cat_prior))  # coders' habit in the category
    return F

def top_codes(r, n):
    s = r.score.values + sum(r[c].values for c in ("score_d", "score_b", "dict_q") if c in r.columns)
    return list(r.code.values[np.argsort(-s, kind="stable")[:n]])

def add_coupling(F, q, res_other, pairs, system, own_col, other_col, loo, grp_pairs=None):
    """Cross-system features. For each candidate: how often it co-occurred in TRAINING with the other system's
    best retrieved codes, and whether the official map (if loaded) links them. Prediction stays independent per system;
    disagreement just becomes information for the ranker."""
    tops = [top_codes(r, 5) for r in res_other]
    g_own = q[own_col].values if loo else None
    g_oth = q[other_col].values if loo else None
    use_grp = loo and GROUP_LOO and grp_pairs is not None and "grp" in q.columns
    g_grp = q["grp"].values if use_grp else None
    qid, code = F.qid.values, F.code.values
    n = len(F); c1 = np.zeros(n); cm = np.zeros(n); cs = np.zeros(n); mp = np.zeros(n)
    for i in range(n):
        O, c = tops[qid[i]], code[i]
        cnt = []
        for o in O:
            key = (o, c) if system == "icd" else (c, o)
            v = pairs.get(key, 0)
            if use_grp: v -= grp_pairs.get((g_grp[qid[i]],) + key, 0)        # v8: remove every pair of the row's own chunk
            elif loo and o == g_oth[qid[i]] and c == g_own[qid[i]]: v -= 1     # remove the row's own pair
            cnt.append(max(v, 0))
        if cnt: c1[i], cm[i], cs[i] = cnt[0], max(cnt), sum(cnt)
        if XMAP_FWD:
            if system == "icd": mp[i] = float(any(c in XMAP_FWD.get(o, ()) for o in O[:3]))
            else: mp[i] = float(any(c in XMAP_REV.get(o, ()) for o in O[:3]))
    F = F.copy()
    F["cx_top1"], F["cx_max"], F["cx_sum"], F["cx_map"] = c1, cm, cs, mp
    return F

def add_llm_pick(F, picks, shown):
    """v8 (teammate Phase 2): llm_pick = 1 if the LLM chose this candidate, 0 if it chose another one, NaN if it gave no answer;
    llm_shown = 1 if the candidate was in the list shown to the LLM."""
    F = F.copy()
    pk = np.array([picks[q] if picks[q] is not None else None for q in F.qid.values], dtype=object)
    has = np.array([p is not None for p in pk])
    F["llm_pick"] = np.where(has, (F.code.values == pk).astype(float), np.nan)
    F["llm_shown"] = [float(c in shown[q]) for q, c in zip(F.qid.values, F.code.values)]
    return F

SECRET_NAMES = {"mistral": "MISTRAL_API_KEY"}

def get_key(provider):
    name = SECRET_NAMES[provider]; k = None
    try:                                   # Colab secrets
        from google.colab import userdata
        k = userdata.get(name)
    except Exception:
        pass
    if not k:
        try:                               # Kaggle secrets (Add-ons -> Secrets, tick "attached")
            from kaggle_secrets import UserSecretsClient
            k = UserSecretsClient().get_secret(name)
        except Exception:
            pass
    return k or os.environ.get(name)

class LLM:
    """Mistral chat client for the LLM pick feature: temperature 0, fixed seed, every answer cached on disk (re-runs are free
    and identical), global requests-per-second limit, backoff on 429 / 5xx."""
    URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, provider, model, rpm=60, cache_path=None):
        import requests
        self.requests, self.provider, self.model = requests, provider, model
        self.key = get_key(provider)
        if not self.key:
            raise RuntimeError(f"No API key: add {SECRET_NAMES[provider]} under Kaggle Add-ons -> Secrets (tick it for this notebook) "
                               "or set the environment variable.")
        self.calls, self.hits, self.max_calls, self.exhausted = 0, 0, None, False
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}

    def save(self):
        if self.cache_path: json.dump(self.cache, open(self.cache_path, "w"))

    def _post(self, prompt):
        return self.requests.post(
            self.URL, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0, "max_tokens": 256, "random_seed": SEED,
                  "messages": [{"role": "user", "content": prompt}]}, timeout=90)

    def _extract(self, j):
        return j["choices"][0]["message"]["content"]

    def ask_many(self, prompts, rps=2.5, workers=8, label="LLM"):
        """v8: many prompts in parallel under a global requests-per-second limit (teammate Phase 2). Cached like ask()."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        key = lambda p: hashlib.sha256(f"{self.provider}|{self.model}|batch|{p}".encode()).hexdigest()
        out = [self.cache.get(key(p)) for p in prompts]
        todo = [i for i, a in enumerate(out) if a is None]
        if not todo: return out
        if self.exhausted or (self.max_calls is not None and self.calls >= self.max_calls): return out
        lock, nxt, t0 = threading.Lock(), [time.time()], time.time()
        def turn():
            with lock:
                now = time.time(); t = max(now, nxt[0]); nxt[0] = t + 1.0 / rps
            if t > now: time.sleep(t - now)
        def one(i):
            if FEATURE_CALL_OVERRIDE is not None: return i, FEATURE_CALL_OVERRIDE(prompts[i])
            for attempt in range(7):
                turn()
                try:
                    r = self._post(prompts[i])
                except Exception:
                    time.sleep(2 ** attempt); continue
                if r.status_code == 200: return i, self._extract(r.json())
                if r.status_code in (429, 500, 502, 503, 504): time.sleep(min(60, 2 ** (attempt + 1))); continue
                print(f"{label} error", r.status_code, r.text[:150]); return i, None
            return i, None
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            for i, a in ex.map(one, todo):
                out[i] = a; done += 1
                with lock:
                    self.calls += 1
                    if a is not None: self.cache[key(prompts[i])] = a
                if done % 200 == 0:
                    log(f"  {label}: {done}/{len(todo)} API calls ({time.time() - t0:.0f}s)"); self.save()
        self.save()
        log(f"  {label}: {len(prompts) - len(todo)} cached, {len(todo)} API calls, {sum(out[i] is None for i in todo)} failed")
        return out

def context_snippet(entity, chunk, width=350):
    c = chunk or ""
    i = c.lower().find(str(entity).lower())
    if i < 0: return c[:2 * width]
    return c[max(0, i - width): i + len(entity) + width]

# ---------------------------------------------------------------------------------------------------------------
# v8 (teammate Phase 2): ONE request per term covering both systems; the pick becomes a reranker feature.
FEAT_SYS_MSG = ("You are an expert clinical coder. For the clinical entity marked << >> in the text, pick the single best "
                "ICD-10-CM option and the single best SNOMED CT option. Follow the coding conventions shown in the examples "
                "(they come from the same annotators). Reply with exactly: ICD=<letter> SNOMED=<letter>")
_LET = "ABCDEFGH"

def feature_prompt(entity, chunk, opts, examples, width=120):
    """opts: {system: [(code, text), ...]} (codes sorted, so the prompt - and its cache entry - only depends on the set)."""
    c = str(chunk or "").replace("<br>", " ")
    i = c.lower().find(str(entity).lower())
    ctx = (c[max(0, i - width):i] + " << " + c[i:i + len(str(entity))] + " >> " + c[i + len(str(entity)):i + len(str(entity)) + width]) \
          if i >= 0 else f"<< {entity} >>"
    lines = [FEAT_SYS_MSG, "", "Examples:"] + [f'- "{e}" -> {s}: {cd} {t}' for e, s, cd, t in examples] + ["", f"Text: {' '.join(ctx.split())}", ""]
    for s, name in (("icd", "ICD-10-CM"), ("snomed", "SNOMED CT")):
        lines.append(f"{name} options:")
        lines += [f"{_LET[j]}) {cd + ' ' if s == 'icd' else ''}{t[:90]}" for j, (cd, t) in enumerate(opts[s])]
        lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)

def parse_letters(ans):
    """(icd_letter, snomed_letter); tolerant of 'ICD=A SNOMED=B', 'ICD: (A), SNOMED: B', 'A, B'."""
    if not ans: return None, None
    up = ans.upper()
    mi = re.search(r"ICD(?:-?10(?:-?CM)?)?[^A-Z0-9]{0,6}\(?([A-H])\b", up)
    ms = re.search(r"SNOMED(?:\s*CT)?[^A-Z0-9]{0,6}\(?([A-H])\b", up)
    if mi and ms: return mi.group(1), ms.group(1)
    both = re.findall(r"\b([A-H])\b", up)
    if len(both) == 2: return (mi.group(1) if mi else both[0]), (ms.group(1) if ms else both[1])
    return (mi.group(1) if mi else None), (ms.group(1) if ms else None)

FEATURE_CLIENT = None
def feature_client():
    """Mistral client for the LLM feature, or None (no key / switched off)."""
    global FEATURE_CLIENT
    if not USE_LLM_FEATURE: return None
    if FEATURE_CLIENT is None:
        if FEATURE_CALL_OVERRIDE is None and not get_key(FEATURE_PROVIDER):
            log(f"LLM feature OFF: no {SECRET_NAMES[FEATURE_PROVIDER]} found"); return None
        if FEATURE_CALL_OVERRIDE is not None and not get_key(FEATURE_PROVIDER):
            os.environ[SECRET_NAMES[FEATURE_PROVIDER]] = "test-override"
        FEATURE_CLIENT = LLM(FEATURE_PROVIDER, FEATURE_MODEL, rpm=60 * FEATURE_RPS, cache_path=FEATURE_CACHE)
    return FEATURE_CLIENT

import copy, traceback, torch
from sklearn.model_selection import GroupKFold

def tuner_texts(df):
    """Text fed to the tuned query encoder: the prepared term, optionally with a slice of its chunk."""
    if TUNE_CTX_CHARS <= 0: return list(df.q_raw.values)
    return [f"{q} | {context_snippet(e, c, TUNE_CTX_CHARS)}" for q, e, c in zip(df.q_raw.values, df.entity.values, df.chunk_text.values)]

import pickle
CKPT_DIR = f"{WORK_DIR}/ckpt"; os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_key(system, df, col, tag, extra=""):
    """Checkpoint name: changes whenever the data or any setting that affects the result changes."""
    h = hashlib.md5()
    h.update(f"{system}|{tag}|{SEED}|{K}|{K_EXTRA}|{USE_BIOLORD}|{USE_TUNED}|{TUNE_EPOCHS}|{TUNE_LR}|{TUNE_BS}|{TUNE_TAU}|"
             f"{TUNE_FOLDS}|{TUNE_CTX_CHARS}|{GROUP_LOO}|{USE_QTY_STRIP}|{CE_DESC if tag == 'ce' else ''}|{CE_DESC_MAX}|{CE_BASE_NAME if tag == 'ce' else ''}|{CE_N}|{CE_FOLDS}|{CE_EPOCHS}|{CE_GROUP}|{CE_BATCHQ}|{CE_LR}|{CE_CTX_CHARS}|{extra}".encode())
    for r, g, q in zip(df.row_id.values, df[col].values, df.q_raw.values): h.update(f"{r}:{g}:{q};".encode())
    return h.hexdigest()[:16]

_BASE = {}      # one shared copy of each pretrained starting point (instead of one per system and per class)

def _scaler():
    try: return torch.amp.GradScaler("cuda")
    except Exception: return torch.cuda.amp.GradScaler()

def _sched(opt, steps):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))


class QueryTuner:
    """A copy of SapBERT that learns to embed a QUERY close to the (frozen) dictionary vectors of its gold code.
    Loss = softmax over ALL dictionary rows, so every other code acts as a negative."""
    def __init__(self, ix):
        from transformers import AutoModel
        self.dev, self.tok, self.ix = ENCODER.dev, ENCODER.tok, ix
        self.max_len = 32 if TUNE_CTX_CHARS <= 0 else 96
        torch.manual_seed(SEED)
        self.m = AutoModel.from_pretrained(SAPBERT_NAME).to(self.dev)
        if "tuner" not in _BASE: _BASE["tuner"] = {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items()}
        self.base = _BASE["tuner"]
        self.Md = ix.Md if torch.is_tensor(ix.Md) else torch.from_numpy(ix.Md).to(self.dev).half()
        counts = np.diff(np.append(ix.starts, len(ix.d)))
        self.row_code = torch.from_numpy(np.repeat(np.arange(len(ix.codes)), counts)).to(self.dev)

    def fit(self, texts, golds):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        self.m.load_state_dict(self.base); self.m.train()
        gp = np.array([self.ix.code_pos.get(c, -1) for c in golds])
        keep = np.where(gp >= 0)[0]
        texts = [texts[i] for i in keep]; gp = gp[keep]
        n = len(texts); steps = TUNE_EPOCHS * math.ceil(n / TUNE_BS)
        opt = torch.optim.AdamW(self.m.parameters(), lr=TUNE_LR, weight_decay=0.01)
        sched, scaler = _sched(opt, steps), _scaler()
        rng = np.random.RandomState(SEED)
        for ep in range(TUNE_EPOCHS):
            perm = rng.permutation(n)
            for i in range(0, n, TUNE_BS):
                idx = perm[i:i + TUNE_BS]
                t = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(self.dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    h = self.m(**t).last_hidden_state[:, 0]
                q = torch.nn.functional.normalize(h.float(), dim=1)
                logits = (q.half() @ self.Md.T).float() / TUNE_TAU
                pos = self.row_code[None, :] == torch.from_numpy(gp[idx]).to(self.dev)[:, None]
                loss = (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~pos, -1e4), 1)).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        self.m.eval()

    def save(self, path): torch.save(self.m.state_dict(), path)
    def load(self, path): self.m.load_state_dict(torch.load(path, map_location=self.dev)); self.m.eval()

    def embed(self, texts):
        self.m.eval()
        texts = [t if t else " " for t in texts]
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                idx = order[i:i + 256]
                t = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(self.dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    h = self.m(**t).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu().numpy())
        E = np.concatenate(out); R = np.empty_like(E); R[order] = E
        return R.astype(np.float32)


def ce_query_text(entity, chunk):
    return f"{entity} | {context_snippet(entity, chunk, CE_CTX_CHARS)}"


class CrossEncoder:
    """BERT-style pair scorer: (term + chunk context, candidate description) -> relevance.
    Trained with a listwise softmax: the gold code against hard negatives from the retrieval shortlist.
    v6: starts from CE_BASE_NAME (a pretrained biomedical reranker) instead of a randomly initialised head on SapBERT."""
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dev = ENCODER.dev
        torch.manual_seed(SEED)
        name = CE_BASE_NAME
        try:
            self.tok = AutoTokenizer.from_pretrained(name)
            self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(self.dev)
        except Exception as e:
            log(f"cross-encoder base {name} unavailable ({e!r:.120}); falling back to SapBERT")
            name = SAPBERT_NAME; self.tok = ENCODER.tok
            self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(self.dev)
        if "ce" not in _BASE:
            _BASE["ce"] = {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items()}
            log(f"cross-encoder base: {name}")
        if "ce" not in _BASE: _BASE["ce"] = {k: v.detach().cpu().clone() for k, v in self.m.state_dict().items()}
        self.base = _BASE["ce"]

    def _tok(self, a, b):
        return self.tok(a, b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.dev)

    def fit(self, pool):
        """pool: list of (query_text, gold_description, [negative descriptions, hardest first])."""
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        self.m.load_state_dict(self.base); self.m.train()
        rng = np.random.RandomState(SEED)
        n = len(pool); steps = CE_EPOCHS * math.ceil(n / CE_BATCHQ)
        opt = torch.optim.AdamW(self.m.parameters(), lr=CE_LR, weight_decay=0.01)
        sched, scaler = _sched(opt, steps), _scaler()
        for ep in range(CE_EPOCHS):
            perm = rng.permutation(n)
            for i in range(0, n, CE_BATCHQ):
                A, B = [], []
                for j in perm[i:i + CE_BATCHQ]:
                    q, pos, negs = pool[j]
                    pick = rng.choice(len(negs), size=CE_GROUP - 1, replace=len(negs) < CE_GROUP - 1)
                    A += [q] * CE_GROUP; B += [pos] + [negs[p] for p in pick]
                t = self._tok(A, B)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.m(**t).logits.squeeze(-1)
                logits = out.float().view(-1, CE_GROUP)
                loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=self.dev))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        self.m.eval()

    def save(self, path): torch.save(self.m.state_dict(), path)
    def load(self, path): self.m.load_state_dict(torch.load(path, map_location=self.dev)); self.m.eval()

    def score(self, A, B, bs=256):
        self.m.eval()
        out = np.zeros(len(A), np.float32)
        order = np.argsort([len(a) + len(b) for a, b in zip(A, B)], kind="stable")
        with torch.no_grad():
            for i in range(0, len(A), bs):
                idx = order[i:i + bs]
                t = self._tok([A[j] for j in idx], [B[j] for j in idx])
                with torch.autocast("cuda", dtype=torch.float16):
                    out[idx] = self.m(**t).logits.squeeze(-1).float().cpu().numpy()
        return out

CE_COLS = ("ce", "ce_gap", "ce_rank", "ce_prob")

def _fused(r):
    """Simple fused retrieval score (sum of every channel's score): picks the cross-encoder's shortlist."""
    return r.score.values + sum(r[c].values for c in ("score_d", "score_b", "dict_q") if c in r.columns)


class SystemModel:
    def __init__(self, system):
        self.system = system
        self.col = "icd10cm_code" if system == "icd" else "snomed_code"
        self.codes, self.raws, self.texts = DICT_ARRAYS[system]
        self.tm = ICD_TEXT if system == "icd" else SN_TEXT
        self.tuner, self.oof = None, None
        self.ce_model, self.model_ce, self.model_base, self.use_ce = None, None, None, False

    def fit_index(self, train_df):
        g = train_df[train_df[self.col] != ""]
        bad = ~g[self.col].isin(VALID_CODES[self.system])                 # v8: gold codes outside the dictionary are not learnable
        if bad.any(): log(f"{self.system}: {int(bad.sum())} training rows have a gold code outside the dictionary (left out of the memory)")
        g = g[~bad].reset_index(drop=True)
        self.rid2ent = dict(zip(g.row_id.astype(int).values, g.entity.values))
        grp = g.grp.values if "grp" in g.columns else None
        self.ix = Index(self.codes, self.texts, self.raws, g[self.col].values, g.q_stem.values, g.q_raw.values,
                        g.row_id.astype(int).values, encoder=ENCODER, enc_b=ENCODER_B, syn_groups=grp)
        self.fz = Featurizer(self.ix, self.system, g, self.col, TAGMAP)

    # ---------- tuned SapBERT: out-of-fold query vectors for train rows, a full model for new data ----------
    def tune(self, train_df):
        self.tuner = None
        if not (USE_TUNED and ENCODER is not None and ENCODER.dev == "cuda"):
            log(f"{self.system}: tuned channel OFF (needs USE_TUNED and a GPU)"); return
        try:
            tr = train_df.reset_index(drop=True)
            texts = tuner_texts(tr); gold = tr[self.col].values
            has = np.where(gold != "")[0]
            groups = pd.factorize(tr.chunk_text.values[has])[0]
            key = ckpt_key(self.system, tr, self.col, "tune")
            p_mod, p_vec = f"{CKPT_DIR}/tune_{self.system}_{key}.pt", f"{CKPT_DIR}/tune_{self.system}_{key}.npy"
            tn = QueryTuner(self.ix)
            if os.path.exists(p_mod) and os.path.exists(p_vec):
                tn.load(p_mod); oof = np.load(p_vec)
                log(f"{self.system}: tuned encoder restored from checkpoint")
            else:
                oof = np.zeros((len(tr), tn.Md.shape[1]), np.float32)
                for k, (a, b) in enumerate(GroupKFold(n_splits=TUNE_FOLDS).split(has, groups=groups)):
                    tn.fit([texts[i] for i in has[a]], list(gold[has[a]]))
                    oof[has[b]] = tn.embed([texts[i] for i in has[b]])
                    log(f"{self.system}: tuned fold {k + 1}/{TUNE_FOLDS}")
                tn.fit([texts[i] for i in has], list(gold[has]))
                no = np.setdiff1d(np.arange(len(tr)), has)
                if len(no): oof[no] = tn.embed([texts[i] for i in no])
                tn.save(p_mod); np.save(p_vec, oof)            # vectors last: both files exist only if the save finished
            self.tuner, self.oof = tn, oof
        except Exception:
            log(f"{self.system}: tuning failed, continuing without it\n" + traceback.format_exc()[-600:])
            self.tuner = None
        torch.cuda.empty_cache()

    # ---------- v5: ICD leaf descent ----------
    def _leaf_inject(self, r, top=30):
        """For each parent/category code among the best `top` candidates, add its default billable child (if missing),
        inheriting the parent's retrieval scores (inh=1). dcp = best score of a parent whose default child is this code."""
        r = r.reset_index(drop=True)
        r["inh"] = 0
        present = set(r.code.values)
        dcp, add = {}, []
        for i in range(min(top, len(r))):
            p = r.code.values[i]; c = ICD_DEFAULT_CHILD.get(p)
            if c is None or c not in self.ix.code_pos: continue
            dcp[c] = max(dcp.get(c, 0.0), float(r.score.values[i]))
            if c not in present:
                row = r.iloc[i].copy(); row["code"] = c; row["inh"] = 1
                add.append(row); present.add(c)
        if add:
            r = pd.concat([r, pd.DataFrame(add)], ignore_index=True)
            r = r.iloc[np.argsort(-r.score.values.astype(float), kind="stable")].reset_index(drop=True)
        r["inh"] = r["inh"].astype(int)
        r["dcp"] = [dcp.get(c, 0.0) for c in r.code.values]
        return r

    # ---------- v5: nearest training terms (few-shot examples for the LLM) ----------
    def neighbours(self, q_stem, q_raw, k, exclude_group=None):
        ix = self.ix
        if k <= 0 or len(ix.syn) == 0: return []
        Qw, Qc = ix.wv.transform([q_stem]), ix.cv.transform([q_stem])
        s = 0.5 * (ix.Sw @ Qw.T).toarray().ravel() + 0.5 * (ix.Sc @ Qc.T).toarray().ravel()
        if ix.encoder is not None and isinstance(getattr(ix, "Sd", None), np.ndarray):
            s = 0.5 * s + 0.5 * (ix.Sd @ ix.encoder([q_raw])[0])
        out, seen = [], set()
        if exclude_group is not None: s = np.where(ix.syn_grp == exclude_group, -np.inf, s)    # v8: never the row's own chunk
        for j in np.argsort(-s, kind="stable")[:5 * k]:
            if not np.isfinite(s[j]): break
            c = ix.syn.code.values[j]
            e = self.rid2ent.get(int(ix.syn.rowid.values[j]), ix.syn.raw.values[j])
            if (e.lower(), c) in seen: continue
            seen.add((e.lower(), c)); out.append((e, c, self.tm.get(c, "")))
            if len(out) >= k: break
        return out

    # ---------- cross-encoder ----------
    def cand_text(self, code, qv=None):
        """v8 (teammate Phase 1b/1c): the code's descriptions ordered by similarity to the term, joined with ' | '
        (up to CE_DESC_MAX characters). qv = the term's SapBERT vector; None -> the code's first description (v7)."""
        title = self.tm.get(code, "")
        if CE_DESC != "mention" or qv is None or code not in self.fz.span: return title
        s, e = self.fz.span[code]
        texts = self.ix.d.raw.values[s:e]
        if e - s > 1:
            sims = self.ix._mm(self.ix.Md[s:e], qv[None, :])[:, 0]
            texts = texts[np.lexsort((np.arange(e - s), -sims))]
        out = ""
        for t in texts:
            t = str(t).strip()
            if not t: continue
            nxt = t if not out else out + " | " + t
            if len(nxt) > CE_DESC_MAX and out: break
            out = nxt[:CE_DESC_MAX]
        return out or title

    def _ce_fill(self, model, qt, res, top, rows, ce, qv=None):
        A, B, where = [], [], []
        for i in rows:
            codes = res[i].code.values
            for k in top[i]:
                A.append(qt[i]); B.append(self.cand_text(codes[k], None if qv is None else qv[i])); where.append((i, k))
        if not A: return
        sc = model.score(A, B)
        for (i, k), v in zip(where, sc): ce[i][k] = v

    def _ce_prepare(self, df, res):
        qt = [ce_query_text(e, c) for e, c in zip(df.entity.values, df.chunk_text.values)]
        top = [np.argsort(-_fused(r), kind="stable")[:CE_N] for r in res]
        ce = [np.full(len(r), np.nan, np.float32) for r in res]
        qv = ENCODER(list(df.q_raw.values)) if (CE_DESC == "mention" and ENCODER is not None and len(df)) else None
        return qt, top, ce, qv

    def ce_fit_oof(self, df, res):
        """Train rows: score with fold models that never saw the row. Also fits a full model for dev/test."""
        df = df.reset_index(drop=True)
        qt, top, ce, qv = self._ce_prepare(df, res)
        gold = df[self.col].values; has = np.where(gold != "")[0]
        pool = {}
        for i in has:
            codes = res[i].code.values
            qvi = None if qv is None else qv[i]
            negs = [self.cand_text(codes[k], qvi) for k in top[i] if codes[k] != gold[i]]
            negs = [t for t in negs if t][:20]
            pos = self.cand_text(gold[i], qvi) if gold[i] in VALID_CODES[self.system] else ""
            if pos and negs: pool[i] = (qt[i], pos, negs)
        groups = pd.factorize(df.chunk_text.values[has])[0]
        key = ckpt_key(self.system, df, self.col, "ce", extra=str(sum(len(r) for r in res)))
        p_mod, p_ce = f"{CKPT_DIR}/ce_{self.system}_{key}.pt", f"{CKPT_DIR}/ce_{self.system}_{key}.pkl"
        model = CrossEncoder()
        if os.path.exists(p_mod) and os.path.exists(p_ce):
            model.load(p_mod); ce = pickle.load(open(p_ce, "rb"))
            log(f"{self.system}: cross-encoder restored from checkpoint")
        else:
            for k, (a, b) in enumerate(GroupKFold(n_splits=CE_FOLDS).split(has, groups=groups)):
                model.fit([pool[i] for i in has[a] if i in pool])
                self._ce_fill(model, qt, res, top, has[b], ce, qv)
                log(f"{self.system}: cross-encoder fold {k + 1}/{CE_FOLDS}")
            model.fit([pool[i] for i in has if i in pool])
            model.save(p_mod); pickle.dump(ce, open(p_ce, "wb"))
        self.ce_model = model
        for i in range(len(res)): res[i]["ce"] = ce[i]
        torch.cuda.empty_cache()

    def ce_score(self, df, res):
        df = df.reset_index(drop=True)
        qt, top, ce, qv = self._ce_prepare(df, res)
        self._ce_fill(self.ce_model, qt, res, top, range(len(res)), ce, qv)
        for i in range(len(res)): res[i]["ce"] = ce[i]

    # ---------- retrieval / features / reranker ----------
    def retrieve(self, df, loo):
        rid = df.row_id.astype(int).values if loo else None
        qv = None
        if self.tuner is not None:
            qv = self.oof if loo else self.tuner.embed(tuner_texts(df))
        ctx = [ctx_text(e, c) for e, c in zip(df.entity.values, df.chunk_text.values)] if USE_CTX_DENSE else None
        cue_s = cue_r = None
        if USE_CUES and "cue_neg" in df.columns:                      # v7: cue-aware search, only for context cues
            pre = [cue_prefix(a, b, c, d) for a, b, c, d in zip(df.cue_neg.values, df.cue_hist.values,
                                                                  df.cue_fam.values, df.cue_in_entity.values)]
            cue_r = [f"{p} {q}" if p else "" for p, q in zip(pre, df.q_raw.values)]
            cue_s = [stem_text(t) for t in cue_r]
        grp = df.grp.values if (loo and "grp" in df.columns) else None
        res = self.ix.retrieve(list(df.q_stem.values), raw_texts=list(df.q_raw.values),
                               query_rowids=rid, K=K, batch=16, qvec=qv, ctx_raws=ctx, cue_stems=cue_s, cue_raws=cue_r,
                               query_groups=grp)
        if self.system == "icd" and ICD_LEAF_FIX:
            res = [self._leaf_inject(r) for r in res]
        if USE_CE and ENCODER is not None and ENCODER.dev == "cuda" and (loo or self.ce_model is not None):
            try:
                if loo: self.ce_fit_oof(df, res)
                else: self.ce_score(df, res)
            except Exception:
                log(f"{self.system}: cross-encoder failed, continuing without it\n" + traceback.format_exc()[-600:])
                self.ce_model = None
                for r in res:
                    if "ce" in r.columns: r.drop(columns="ce", inplace=True)
        return res

    def features(self, df, res, res_other, pairs, loo, with_gold, grp_pairs=None, picks=None):
        q = df.reset_index(drop=True).copy()
        gcol = self.col
        if not with_gold:
            q["_gold"] = ""; gcol = "_gold"
        F = self.fz.build(q, "q_stem", "chunk_text", gcol, res, loo=loo, orig_col="entity")
        if USE_COUPLING:
            other = "snomed_code" if self.system == "icd" else "icd10cm_code"
            F = add_coupling(F, q, res_other, pairs, self.system, self.col, other, loo and with_gold, grp_pairs)
        if picks is not None:                                           # v8: LLM pick feature
            F = add_llm_pick(F, picks[0], picks[1])
        if len(res) and "ce" in res[0].columns:
            F = F.copy()
            F["ce"] = np.concatenate([r["ce"].values for r in res])
            g = F.groupby("qid")["ce"]
            mx = g.transform("max")
            F["ce_gap"] = mx - F["ce"]
            F["ce_rank"] = g.rank(ascending=False, method="first")
            e = np.exp(F["ce"] - mx)
            F["ce_prob"] = e / e.groupby(F["qid"]).transform("sum")
        if SIBLING_FEATS:
            F = add_sibling_features(F, self.system)
        return F

    def _fit_rank(self, F, feats, rounds=300):
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=rounds, learning_rate=0.05, num_leaves=15,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
        model.fit(F[feats], F.label, group=F.groupby("qid", sort=False).size().values)
        return model

    def _choose_rounds(self, F, feats, groups):
        """v8 (teammate round curve): top-1 accuracy of chunk-grouped CV folds at each RANKER_ROUNDS checkpoint."""
        if groups is None or not RANKER_ROUNDS or not RANKER_CV: return 300, {}
        qids = np.sort(F.qid.unique())
        acc = {k: [] for k in RANKER_ROUNDS}
        for a, b in GroupKFold(n_splits=RANKER_CV).split(qids, groups=groups[qids]):
            tr, te = F[F.qid.isin(qids[a])], F[F.qid.isin(qids[b])]
            m = self._fit_rank(tr, feats, max(RANKER_ROUNDS))
            for k in RANKER_ROUNDS:
                p = te.assign(_p=m.predict(te[feats], num_iteration=k))
                acc[k].append(float(p.loc[p.groupby("qid")._p.idxmax()].label.mean()))
        curve = {k: round(100 * float(np.mean(v)), 2) for k, v in acc.items()}
        return max(curve, key=lambda k: (curve[k], -k)), curve

    def train_ranker(self, F, groups=None):
        allf = [c for c in F.columns if c not in ("qid", "code", "label")]
        self.feats_ce = allf
        self.feats_base = [c for c in allf if c not in CE_COLS]
        self.rounds = {}
        for name, feats in (("base", self.feats_base), ("ce", self.feats_ce if "ce" in F.columns else None)):
            if feats is None: continue
            r, curve = self._choose_rounds(F, feats, groups)
            self.rounds[name] = r
            if curve: log(f"{self.system} ({name}): grouped-CV top-1 % by boosting rounds {curve} -> {r}")
        self.model_base = self._fit_rank(F, self.feats_base, self.rounds["base"])
        self.model_ce = self._fit_rank(F, self.feats_ce, self.rounds["ce"]) if "ce" in F.columns else None

    def score(self, F, use_ce=None):
        use = self.use_ce if use_ce is None else use_ce
        use = bool(use) and self.model_ce is not None and "ce" in F.columns
        feats, model = (self.feats_ce, self.model_ce) if use else (self.feats_base, self.model_base)
        F = F.copy()
        for c in feats:
            if c not in F.columns: F[c] = np.nan
        F["pred"] = model.predict(F[feats])
        e = np.exp(F.pred - F.groupby("qid").pred.transform("max"))
        F["prob"] = e / e.groupby(F.qid).transform("sum")       # softmax over each query's candidates = confidence
        return F


def _pre_rank(r):
    """order used to choose the candidates shown to the LLM feature: cross-encoder score where present, then fused retrieval."""
    f = _fused(r)
    if "ce" in r.columns:
        c = r["ce"].values.astype(float)
        return np.lexsort((-f, -np.where(np.isnan(c), -1e9, c)))
    return np.argsort(-f, kind="stable")


class Joint:
    def __init__(self):
        self.m = {"icd": SystemModel("icd"), "snomed": SystemModel("snomed")}
        self.pairs, self.grp_pairs = {}, {}
        self._last = {}
        self.use_llm_feat = False

    def llm_feature_picks(self, df, res, loo):
        """v8: one LLM request per term (both systems). Returns {system: (picks per row, shown codes per row)} or None.
        Training rows (loo) only see examples from OTHER chunks."""
        if not self.use_llm_feat: return None
        cl = feature_client()
        if cl is None: return None
        df = df.reset_index(drop=True); n = len(df)
        opts, prompts = [], []
        for i in range(n):
            o = {}
            for s, m in self.m.items():
                top = res[s][i].code.values[_pre_rank(res[s][i])[:FEATURE_TOP_N]]
                o[s] = [(c, m.tm.get(c, "")) for c in sorted(top)]
            ex = []
            g = df.grp.values[i] if (loo and "grp" in df.columns) else None      # never an example from the row's own chunk
            for s, m in self.m.items():
                for e, c, t in m.neighbours(df.q_stem.values[i], df.q_raw.values[i], FEATURE_EX, exclude_group=g):
                    ex.append((e, "ICD" if s == "icd" else "SNOMED", c if s == "icd" else "", t))
            opts.append(o); prompts.append(feature_prompt(df.entity.values[i], df.chunk_text.values[i], o, ex))
        answers = cl.ask_many(prompts, rps=FEATURE_RPS, label="LLM feature")
        out = {s: ([None] * n, [set(c for c, _ in o[s]) for o in opts]) for s in self.m}
        for i, a in enumerate(answers):
            for s, letter in zip(("icd", "snomed"), parse_letters(a)):
                if letter is not None and _LET.index(letter) < len(opts[i][s]):
                    out[s][0][i] = opts[i][s][_LET.index(letter)][0]
        got = sum(p is not None for p in out["icd"][0])
        log(f"LLM feature: picks for {got}/{n} terms")
        return out

    def set_ce(self, flags):
        """flags: True/False for both systems, or {"icd": bool, "snomed": bool}."""
        for s, m in self.m.items():
            f = flags if isinstance(flags, bool) else flags.get(s, False)
            m.use_ce = bool(f) and m.model_ce is not None

    def fit(self, train_df):
        train_df = train_df.reset_index(drop=True)
        for m in self.m.values():
            m.fit_index(train_df); m.tune(train_df)
        log("indexes built, tuned encoders trained")
        res = {s: m.retrieve(train_df, loo=True) for s, m in self.m.items()}    # includes cross-encoder cross-fitting
        log("train retrieval done")
        both = train_df[(train_df.snomed_code != "") & (train_df.icd10cm_code != "")]
        self.pairs = collections.Counter(zip(both.snomed_code, both.icd10cm_code))
        if "grp" in both.columns:
            self.grp_pairs = collections.Counter(zip(both.grp, both.snomed_code, both.icd10cm_code))
        self.use_llm_feat = feature_client() is not None                # v8: decided once at fit time, reused when scoring
        picks = self.llm_feature_picks(train_df, res, loo=True)
        for s, m in self.m.items():
            o = "snomed" if s == "icd" else "icd"
            pos = np.where(train_df[m.col].values != "")[0]
            sub = train_df.iloc[pos].reset_index(drop=True)
            pk = None if picks is None else ([picks[s][0][i] for i in pos], [picks[s][1][i] for i in pos])
            F = m.features(sub, [res[s][i] for i in pos], [res[o][i] for i in pos], self.pairs, loo=True, with_gold=True,
                           grp_pairs=self.grp_pairs, picks=pk)
            m.train_ranker(F, groups=sub.grp.values if "grp" in sub.columns else None)
            log(f"{s}: reranker trained on {F.qid.nunique()} queries / {len(F):,} candidates "
                f"({'with' if m.model_ce is not None else 'without'} cross-encoder feature)")
            del F, sub; gc.collect()          # (res stays: the other system still needs it for the coupling features)
        self.set_ce(CE_MODE != "off")
        del res; gc.collect()
        return self

    def score(self, df, use_ce=None):
        """Candidate tables with reranker score + confidence, per system. Never looks at gold columns."""
        t = time.time()
        df = df.reset_index(drop=True)
        res = {s: m.retrieve(df, loo=False) for s, m in self.m.items()}
        picks = self.llm_feature_picks(df, res, loo=False)
        out = {}
        for s, m in self.m.items():
            o = "snomed" if s == "icd" else "icd"
            self._last[s] = m.features(df, res[s], res[o], self.pairs, loo=False, with_gold=False,
                                       picks=None if picks is None else picks[s])
            out[s] = m.score(self._last[s], use_ce)
        self.ms_per_term = (time.time() - t) * 1000 / max(1, len(df))
        return out

    def rescore(self, use_ce):
        """Re-rank the SAME features of the last score() call with / without the cross-encoder feature (no recomputation)."""
        return {s: m.score(self._last[s], use_ce) for s, m in self.m.items()}

    def decide(self, df, scored):
        """Top-1 code per system from the reranker, with its confidence (softmax probability over the term's candidates)."""
        n = len(df); out = pd.DataFrame(index=range(n))
        for s, m in self.m.items():
            F = scored[s]
            top = F.loc[F.groupby("qid").pred.idxmax()].set_index("qid")
            pred = top.code.reindex(range(n)).fillna("").values
            out[m.col + "_ranker"] = pred
            out[s + "_prob"] = top.prob.reindex(range(n)).fillna(0).values
            out[m.col] = pred
        return out


def _acc(ok, mask):
    return float(ok[mask].mean()) if mask.sum() else float("nan")

def evaluate(pred, df, fit_df, title=""):
    """Exact-match accuracy per system (the headline is the LOWER of the two), long-tail split, hierarchical
    partial credit, and both-correct. `pred` has icd10cm_code / snomed_code; `df` holds the gold."""
    rows = []
    for name, col in (("ICD-10-CM", "icd10cm_code"), ("SNOMED CT", "snomed_code")):
        y, p = df[col].values, pred[col].values
        has = y != ""
        seen = np.isin(y, list(set(fit_df[col]) - {""}))
        ok = p == y
        r = {"system": name, "n": int(has.sum()), "EXACT": _acc(ok, has),
             "seen_codes": _acc(ok, has & seen), "n_seen": int((has & seen).sum()),
             "LONG-TAIL (unseen)": _acc(ok, has & ~seen), "n_unseen": int((has & ~seen).sum())}
        if col == "icd10cm_code":
            r["same 3-char category"] = _acc(np.array([a.split(".")[0] for a in p]) == np.array([b.split(".")[0] for b in y]), has)
            r["same chapter letter"] = _acc(np.array([a[:1] for a in p]) == np.array([b[:1] for b in y]), has)
        else:
            r["same semantic tag"] = _acc(np.array([TAGMAP.get(a, "?") for a in p]) == np.array([TAGMAP.get(b, "?") for b in y]), has)
            hi = has & (df.icd10cm_code.values != "")                 # v8: the teammate's population (mentions with ICD gold)
            r["EXACT on mentions with ICD gold"] = _acc(ok, hi); r["n_with_icd"] = int(hi.sum())
        rows.append(r)
    both = (df.icd10cm_code.values != "") & (df.snomed_code.values != "")
    both_ok = (pred.icd10cm_code.values == df.icd10cm_code.values) & (pred.snomed_code.values == df.snomed_code.values)
    T = pd.DataFrame(rows).set_index("system")
    print(f"===== {title} =====")
    print(T.round(3).T.to_string())
    print(f"HEADLINE (lower of the two exact-match accuracies): {T.EXACT.min():.3f}")
    print(f"comparable with the teammate notebook (mentions with ICD gold): "
          f"{min(T.EXACT['ICD-10-CM'], T['EXACT on mentions with ICD gold']['SNOMED CT']):.3f}")
    print(f"both systems correct on the same entity: {_acc(both_ok, both):.3f}  (n={int(both.sum())})\n")
    return T

def stage_predictions(scored, n):
    """Top-1 of each stage, for the ablation table: lexical+memory only / dense only / hybrid / reranker."""
    res = {}
    for s, F in scored.items():
        F = F.copy()
        F["hybrid"] = F.score + sum(F[x] for x in ("score_d", "score_b", "dict_q") if x in F.columns)
        for name, col in (("lexical+memory", "score"), ("dense (SapBERT)", "score_d"), ("tuned SapBERT", "dict_q"),
                          ("BioLORD", "score_b"), ("hybrid", "hybrid"), ("reranker", "pred")):
            if col in F.columns:
                top = F.loc[F.groupby("qid")[col].idxmax()].set_index("qid")
                res[(s, name)] = top.code.reindex(range(n)).fillna("").values
    return res

def ablation(df, stage_preds, final_preds=None):
    """What each stage adds. final_preds: {'+ LLM adjudication': prediction DataFrame, ...}"""
    rows = []
    def one(name, p_icd, p_sn):
        yi, ys = df.icd10cm_code.values, df.snomed_code.values
        a = _acc(p_icd == yi, yi != ""); b = _acc(p_sn == ys, ys != "")
        rows.append({"stage": name, "ICD": a, "SNOMED": b, "headline(min)": min(a, b)})
    for name in ("lexical+memory", "dense (SapBERT)", "tuned SapBERT", "BioLORD", "hybrid", "reranker"):
        if ("icd", name) in stage_preds: one(name, stage_preds[("icd", name)], stage_preds[("snomed", name)])
    for name, P in (final_preds or {}).items(): one(name, P.icd10cm_code.values, P.snomed_code.values)
    print(pd.DataFrame(rows).set_index("stage").round(3).to_string()); print()

def candidate_report(scored, df):
    """Recall of the candidate list and of the reranker's top-k (a miss here can never be fixed later)."""
    n = len(df)
    for s, col in (("icd", "icd10cm_code"), ("snomed", "snomed_code")):
        F = scored[s]; y = df[col].values
        lists = (F.sort_values(["qid", "pred"], ascending=[True, False], kind="stable")
                   .groupby("qid").code.apply(list).reindex(range(n)))
        qs = [q for q in range(n) if y[q] != ""]
        r = {"gold in candidate list": np.mean([y[q] in lists[q] for q in qs])}
        for k in (1, 3, 5, 10): r[f"gold in reranker top-{k}"] = np.mean([y[q] in lists[q][:k] for q in qs])
        print(s.upper(), {k_: round(v, 3) for k_, v in r.items()})
    print()

def confidence_report(df, pred):
    """Accuracy by reranker confidence, and the coverage / abstention trade-off."""
    for s, col in (("icd", "icd10cm_code"), ("snomed", "snomed_code")):
        y = df[col].values; has = y != ""
        ok = pred[col + "_ranker"].values == y; pr = pred[s + "_prob"].values
        print(f"--- {s.upper()}: reranker accuracy by confidence")
        rows = []
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            lo, hi = has & (pr < t), has & (pr >= t)
            rows.append({"threshold": t, "share below": lo.sum() / has.sum(),
                         "acc below": _acc(ok, lo), "acc above": _acc(ok, hi)})
        print(pd.DataFrame(rows).set_index("threshold").round(3).to_string())
        order = np.argsort(-pr[has], kind="stable"); okh = ok[has][order]
        cov = {f"{c:.0%}": round(float(okh[:max(1, int(len(okh) * c))].mean()), 3) for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)}
        print("abstention curve (accuracy on the most confident X% of terms):", cov, "\n")

def error_table(df, pred, scored, fit_df, path):
    """One row per wrong prediction, with a failure category - feeds the error-analysis write-up."""
    n = len(df); rows = []
    for s, col, tm in (("icd", "icd10cm_code", ICD_TEXT), ("snomed", "snomed_code", SN_TEXT)):
        F = scored[s]; y, p = df[col].values, pred[col].values
        lists = (F.sort_values(["qid", "pred"], ascending=[True, False], kind="stable")
                   .groupby("qid").code.apply(list).reindex(range(n)))
        seen = set(fit_df[col]) - {""}
        for q in range(n):
            if y[q] == "" or p[q] == y[q]: continue
            a, b = p[q], y[q]
            if s == "icd":
                kind = ("parent/child (specificity)" if (a and (a.startswith(b) or b.startswith(a)))
                        else "same 3-char category" if a.split(".")[0] == b.split(".")[0] else "different category")
            else:
                kind = "same semantic tag" if TAGMAP.get(a) == TAGMAP.get(b) else "different semantic tag"
            rank = lists[q].index(b) + 1 if b in lists[q] else None
            rows.append({"system": s, "entity": df.entity.values[q], "chunk": (df.chunk_text.values[q] or "")[:300],
                         "pred": a, "pred_text": tm.get(a, ""), "gold": b, "gold_text": tm.get(b, ""),
                         "kind": kind, "gold_in_candidates": rank is not None, "gold_rank": rank,
                         "gold_unseen_in_train": b not in seen})
    E = pd.DataFrame(rows); E.to_csv(path, index=False)
    print(f"{len(E)} errors saved to {path}")
    if len(E):
        print(E.groupby(["system", "kind"]).size().to_string())
        print(E.groupby("system").gold_in_candidates.mean().round(3).rename("share of errors where gold WAS a candidate").to_string())
    return E

def predict_file(in_path, out_path, model=None):
    model = model or globals().get("J_sub") or globals().get("J_final") or globals().get("J")
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    ent = "clinical_entity" if "clinical_entity" in df.columns else "entity"
    q = pd.DataFrame({"entity": df[ent].values,
                      "chunk_text": df["chunk_text"].values if "chunk_text" in df.columns else ""})
    q["row_id"] = -1 - np.arange(len(q))                       # never collides with a training row
    q["grp"] = -1 - np.arange(len(q))
    q = add_query_columns(q)
    p = model.decide(q, model.score(q))
    out = pd.DataFrame({"clinical_entity": df[ent].values})
    if "chunk_id" in df.columns: out.insert(0, "chunk_id", df["chunk_id"].values)
    out["icd10cm_code"], out["snomed_code"] = p.icd10cm_code.values, p.snomed_code.values
    bad = {s: int((~out[c].isin(VALID_CODES[s]) & (out[c] != "")).sum()) for s, c in (("icd", "icd10cm_code"), ("snomed", "snomed_code"))}
    assert not any(bad.values()), f"predicted codes outside the dictionary: {bad}"
    out.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(out)} rows)")
    return out

# predict_file(f"{DATA_DIR}/evaluation_input.csv", f"{OUT_DIR}/predictions.csv")

def demo(items, model=None, show=3):
    """items: list of (entity, chunk_text). Prints the top candidates per system with their reranker probability."""
    model = model or globals().get("J_sub") or globals().get("J_final") or globals().get("J")
    q = pd.DataFrame({"entity": [a for a, b in items], "chunk_text": [b for a, b in items]})
    q["row_id"] = -1 - np.arange(len(q)); q["grp"] = -1 - np.arange(len(q)); q = add_query_columns(q)
    scored = model.score(q)
    p = model.decide(q, scored)
    for i, (ent, chunk) in enumerate(items):
        print(f"\n### {ent!r}   (prepared query: {q.q_raw[i]!r})")
        for s, tm in (("icd", ICD_TEXT), ("snomed", SN_TEXT)):
            g = scored[s][scored[s].qid == i].sort_values("pred", ascending=False).head(show)
            final = p["icd10cm_code" if s == "icd" else "snomed_code"][i]
            print(f"  {s.upper():6s} -> {final}  [{tm.get(final, '')}]  confidence {p[s + '_prob'][i]:.2f}")
            for c, pr in zip(g.code.values, g.prob.values):
                print(f"           {c:>12s}  p={pr:.2f}  {tm.get(c, '')[:70]}")
