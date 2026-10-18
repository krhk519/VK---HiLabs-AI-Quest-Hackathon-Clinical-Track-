"""Unpack hilabs_weights.zip (section 15c of the notebook) so run.py reloads checkpoints instead of retraining.
    python scripts/restore_weights.py hilabs_weights.zip [work_dir]
"""
import re, shutil, sys, zipfile, os, tempfile
zp = sys.argv[1]; work = sys.argv[2] if len(sys.argv) > 2 else "work"
ck = os.path.join(work, "ckpt"); os.makedirs(ck, exist_ok=True)
pat = re.compile(r"^(tune|ce)_(icd|snomed)_[0-9a-f]{16}\.(pt|npy|pkl)$")
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile(zp).extractall(t)
    for f in os.listdir(t):
        if pat.match(f): shutil.copy(os.path.join(t, f), ck)
        elif f == "llm_feature_cache.json": shutil.copy(os.path.join(t, f), work)
print("restored into", work)
