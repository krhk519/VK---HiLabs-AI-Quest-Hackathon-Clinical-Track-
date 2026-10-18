"""Pull the saved checkpoints from the Hugging Face Hub into work/ so run.py reloads them instead of retraining.

    python scripts/download_weights.py <hf_user>/<repo>  [work_dir]      (or set HF_WEIGHTS_REPO)

Works with loose files or an uploaded hilabs_weights.zip, in any folder layout in the HF repo: every file named like tune_icd_<hash>.pt / ce_snomed_<hash>.pkl goes to
work/ckpt, and llm_feature_cache.json goes to work/. A read token (HF_TOKEN) is only needed if the repo is private."""
import os, re, shutil, sys, tempfile, zipfile
from huggingface_hub import snapshot_download

DEFAULT_REPO = "krhk/Hilabs-term-mapping-weights"      # <-- EDIT: your Hugging Face repo id (not a secret), so judges need no setup
repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_WEIGHTS_REPO") or DEFAULT_REPO
work = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HILABS_WORK_DIR", "work")
if not repo or "ADD-" in repo: sys.exit("HF repo id not set: edit DEFAULT_REPO in scripts/download_weights.py")
ck = os.path.join(work, "ckpt"); os.makedirs(ck, exist_ok=True)
src = snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)
pat = re.compile(r"^(tune|ce)_(icd|snomed)_[0-9a-f]{16}\.(pt|npy|pkl)$")
n = 0
def take(folder):
    global n
    for dp, _, fns in os.walk(folder):
        for f in fns:
            if pat.match(f): shutil.copy(os.path.join(dp, f), ck); n += 1
            elif f == "llm_feature_cache.json": shutil.copy(os.path.join(dp, f), work); n += 1
take(src)
for dp, _, fns in os.walk(src):                      # a zip uploaded as-is (e.g. hilabs_weights.zip) is unpacked too
    for f in fns:
        if f.endswith(".zip"):
            with tempfile.TemporaryDirectory() as t:
                zipfile.ZipFile(os.path.join(dp, f)).extractall(t); take(t)
print(f"downloaded {n} files from {repo} into {work}")
if n == 0: sys.exit("no checkpoint files found in that repo")
