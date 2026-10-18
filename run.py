#!/usr/bin/env python
"""One command: fit the engine on ALL labelled rows, then map an evaluation file.

    python run.py predict --input data/evaluation_input.csv --output outputs/predictions.csv
    python run.py lockbox                      # reproduce the held-out (lockbox) score reported in the write-up
    python run.py demo                         # print the hard-case demo (abbreviation, OCR noise, specificity)

Input columns : clinical_entity (or entity), optional chunk_text, optional chunk_id (passed through)
Output columns: chunk_id, clinical_entity, icd10cm_code, snomed_code
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["predict", "lockbox", "demo"])
ap.add_argument("--input"); ap.add_argument("--output", default="outputs/predictions.csv")
ap.add_argument("--data-dir"); ap.add_argument("--work-dir"); ap.add_argument("--out-dir")
ap.add_argument("--no-llm", action="store_true", help="skip the Mistral pick feature (no data leaves the machine)")
a = ap.parse_args()
if a.data_dir: os.environ["HILABS_DATA_DIR"] = a.data_dir
if a.work_dir: os.environ["HILABS_WORK_DIR"] = a.work_dir
if a.out_dir:  os.environ["HILABS_OUT_DIR"] = a.out_dir
if a.no_llm:   os.environ["USE_LLM_FEATURE"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import engine as E                      # loads data, dictionaries, encoders (GPU recommended)
import pandas as pd, gc

def fit(df):
    J = E.Joint().fit(df.reset_index(drop=True))
    J.set_ce(True)                       # cross-encoder feature on (decided on dev during development, see write-up)
    return J

if a.mode == "predict":
    if not a.input: ap.error("--input is required for predict")
    J = fit(pd.concat([E.train, E.dev, E.test], ignore_index=True))   # all labelled rows
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    E.predict_file(a.input, a.output, model=J)
elif a.mode == "lockbox":
    fit_all = pd.concat([E.train, E.dev], ignore_index=True)
    J = fit(fit_all)
    cand = J.score(E.test); pred = J.decide(E.test, cand)
    E.evaluate(pred, E.test, fit_all, "LOCKBOX - final engine")
    E.candidate_report(cand, E.test)
    E.error_table(E.test, pred, cand, fit_all, f"{E.OUT_DIR}/test_errors.csv")
    out = E.test[["entity", "chunk_text", "icd10cm_code", "snomed_code"]].rename(
        columns={"icd10cm_code": "gold_icd10cm_code", "snomed_code": "gold_snomed_code"})
    out["icd10cm_code"], out["snomed_code"] = pred.icd10cm_code.values, pred.snomed_code.values
    out["icd_confidence"], out["snomed_confidence"] = pred.icd_prob.values, pred.snomed_prob.values
    out.to_csv(f"{E.OUT_DIR}/test_predictions.csv", index=False)
else:
    J = fit(pd.concat([E.train, E.dev], ignore_index=True))
    E.demo([("HTN", "Past medical history: HTN, hyperlipidemia."),
            ("pulmonary hypertension", "Echo shows elevated pulmonary pressures consistent with pulmonary hypertension."),
            ("chronickidney disease stag 3", "Assessment: chronickidney disease stag 3, stable creatinine."),
            ("diabetic neuropathy", "Type 2 diabetes with diabetic neuropathy of both feet."),
            ("acute uri", "Presents with 3 days of cough and congestion, likely acute uri."),
            ("chest pain", "Patient denies chest pain, palpitations or syncope."),
            ("MS", "Neuro: relapsing-remitting MS on ocrelizumab, last MRI stable.")], model=J)
