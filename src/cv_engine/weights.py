"""Publish / download trained CV weights via the Hugging Face Hub.

Lets teammates run inference **without a GPU or retraining**: they just
``download()`` the trained flower/insect/classifier weights.

Publishing needs a write token (``hf auth login`` once). Downloading a public
repo needs nothing.

CLI:
    python -m src.cv_engine.weights --download            # teammates: fetch weights
    python -m src.cv_engine.weights --publish             # maintainer: upload (needs token)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config as C

CV_WEIGHTS_DIR = C.INTERIM_DIR / "weights"

# local trained-weight path -> filename in the HF repo (only existing ones upload)
_ARTIFACTS = {
    C.INTERIM_DIR / "cv_runs" / "flower_yolo26n" / "weights" / "best.pt": "flower_yolo26.pt",
    C.INTERIM_DIR / "cv_runs" / "insect1cls_yolo26n" / "weights" / "best.pt": "insect_yolo26.pt",
    C.INTERIM_DIR / "cv_runs" / "insect_classifier" / "best.pt": "insect_classifier.pt",
}


def publish(repo: str = C.HF_WEIGHTS_REPO) -> list[str]:
    """Upload every trained weight that exists to the HF model repo."""
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo, repo_type="model", exist_ok=True)
    uploaded = []
    for local, name in _ARTIFACTS.items():
        if local.exists():
            api.upload_file(path_or_fileobj=str(local), path_in_repo=name,
                            repo_id=repo, repo_type="model")
            uploaded.append(name)
    return uploaded


def download(repo: str = C.HF_WEIGHTS_REPO, dst: Path = CV_WEIGHTS_DIR) -> list[str]:
    """Download all published weights into data/interim/weights/ for inference."""
    from huggingface_hub import list_repo_files, hf_hub_download
    dst.mkdir(parents=True, exist_ok=True)
    got = []
    for f in list_repo_files(repo, repo_type="model"):
        if f.endswith(".pt"):
            hf_hub_download(repo, f, repo_type="model", local_dir=str(dst))
            got.append(f)
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--repo", default=C.HF_WEIGHTS_REPO)
    args = ap.parse_args()
    if args.publish:
        print("uploaded:", publish(args.repo))
    elif args.download:
        print("downloaded:", download(args.repo))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
