#!/usr/bin/env python3
"""Scan ModelScope dataset for missing LFS blobs; re-upload only broken tasks."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from modelscope.hub.api import HubApi

NAMESPACE = "Twilighted"
DATASET = "Robocasa365-tactile"
REPO_ID = f"{NAMESPACE}/{DATASET}"
BATCH_URL = f"https://www.modelscope.cn/api/v1/repos/datasets/{REPO_ID}/info/lfs/objects/batch"
BATCH_SIZE = 100
IGNORE_PARTS = {".ms_upload_cache", "._____temp", ".msc", ".mv"}


def list_remote_blobs(api: HubApi, token: str, task_path: str) -> dict[str, dict]:
    """Return {repo_path: {Sha256, Size}} for blob files under task_path."""
    out: dict[str, dict] = {}
    page = 1
    while True:
        resp = api.list_repo_tree(
            dataset_name=DATASET,
            namespace=NAMESPACE,
            revision="master",
            root_path=task_path,
            recursive=True,
            page_number=page,
            page_size=500,
            token=token,
        )
        files = resp.get("Data", {}).get("Files", [])
        total = resp.get("Data", {}).get("TotalCount", 0)
        for f in files:
            if f.get("Type") != "blob":
                continue
            path = f.get("Path", "")
            sha = f.get("Sha256")
            size = f.get("Size")
            if sha and size is not None:
                out[path] = {"Sha256": sha, "Size": size}
        if page * 500 >= total or not files:
            break
        page += 1
    return out


def lfs_missing_oids(api: HubApi, token: str, objects: list[dict]) -> set[str]:
    """Return oids whose LFS blob is missing (batch returns upload action)."""
    missing: set[str] = set()
    headers = api.builder_headers(api.headers)
    headers["Authorization"] = f"Bearer {token}"
    for i in range(0, len(objects), BATCH_SIZE):
        chunk = objects[i : i + BATCH_SIZE]
        payload = {"operation": "download", "transfers": ["basic"], "objects": chunk}
        r = api.session.post(BATCH_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json().get("Data", {})
        for obj in data.get("objects", []):
            actions = obj.get("actions") or {}
            if "upload" in actions and "download" not in actions:
                missing.add(obj["oid"])
    return missing


def check_task(api: HubApi, token: str, task_path: str) -> tuple[str, int, int, bool]:
    """Return (task_path, total_blobs, missing_count, needs_fix)."""
    remote = list_remote_blobs(api, token, task_path)
    if not remote:
        return task_path, 0, 0, True

    objects = [{"oid": m["Sha256"], "size": m["Size"]} for m in remote.values()]
    missing = lfs_missing_oids(api, token, objects)
    return task_path, len(objects), len(missing), len(missing) > 0


def upload_task(
    token: str,
    local_root: Path,
    task_path: str,
    max_workers: int,
    log_dir: Path,
) -> bool:
    local_path = local_root / task_path
    task_log = log_dir / f"fix_{task_path.replace('/', '_')}.log"
    env = os.environ.copy()
    env["UPLOAD_USE_CACHE"] = "false"
    cmd = [
        "modelscope",
        "upload",
        REPO_ID,
        str(local_path),
        task_path,
        "--repo-type",
        "dataset",
        "--token",
        token,
        "--max-workers",
        str(max_workers),
        "--exclude",
        "**/.ms_upload_cache/**",
        "**/.ms_upload_cache",
        "**/._____temp/**",
        "--commit-message",
        f"fix LFS blob: selective reupload {task_path}",
    ]
    with open(task_log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.returncode == 0


def verify_task_sample(api: HubApi, token: str, task_path: str) -> bool:
    """Quick post-upload check: one small meta file must be downloadable."""
    remote = list_remote_blobs(api, token, task_path)
    candidates = [p for p in remote if p.endswith("meta/info.json")]
    if not candidates:
        candidates = list(remote.keys())[:1]
    if not candidates:
        return False
    path = candidates[0]
    url = api.get_dataset_file_url(
        path, dataset_name=DATASET, namespace=NAMESPACE, revision="master"
    )
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    expected = remote[path]["Size"]
    return len(r.content) == expected and expected > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            Path(__file__).resolve().parents[2] / "../datasets/robocasa365-datasets"
        ),
    )
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_TOKEN", ""))
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    if not args.token:
        print("Set MODELSCOPE_TOKEN", file=sys.stderr)
        return 1

    local_root = Path(args.root).resolve()
    project_root = Path(__file__).resolve().parents[2]
    log_dir = Path(args.log_dir or project_root / "output/modelscope_upload")
    log_dir.mkdir(parents=True, exist_ok=True)

    scan_file = log_dir / "broken_tasks.txt"
    ok_file = log_dir / "ok_tasks.txt"
    progress_file = log_dir / "lfs_selective_fix_progress.txt"

    tasks: list[str] = []
    for split in ("pretrain", "target"):
        for cat in ("atomic", "composite"):
            base = local_root / split / cat
            if not base.is_dir():
                continue
            for task in sorted(base.iterdir()):
                if task.is_dir():
                    tasks.append(f"{split}/{cat}/{task.name}")

    api = HubApi()
    api.login(args.token)

    broken: list[str] = []
    ok: list[str] = []

    if not args.upload_only:
        print(f"[scan] {len(tasks)} tasks, workers={args.scan_workers}")
        t0 = time.time()

        def _check(tp: str) -> tuple[str, int, int, bool]:
            return check_task(api, args.token, tp)

        with ThreadPoolExecutor(max_workers=args.scan_workers) as ex:
            futs = {ex.submit(_check, tp): tp for tp in tasks}
            done = 0
            for fut in as_completed(futs):
                tp, total, miss, needs = fut.result()
                done += 1
                if needs:
                    broken.append(tp)
                    print(f"[broken] {tp} missing={miss}/{total}")
                else:
                    ok.append(tp)
                if done % 20 == 0 or done == len(tasks):
                    print(f"[scan] {done}/{len(tasks)} elapsed={time.time()-t0:.0f}s")

        broken.sort()
        ok.sort()
        scan_file.write_text("\n".join(broken) + ("\n" if broken else ""))
        ok_file.write_text("\n".join(ok) + ("\n" if ok else ""))
        print(f"[scan] broken={len(broken)} ok={len(ok)} -> {scan_file}")
        if args.scan_only:
            return 0

    if args.upload_only:
        if not scan_file.exists():
            print(f"Missing {scan_file}; run scan first", file=sys.stderr)
            return 1
        broken = [l.strip() for l in scan_file.read_text().splitlines() if l.strip()]

    # skip already fixed
    fixed = set()
    if progress_file.exists():
        fixed = {l.split(" ", 1)[1] for l in progress_file.read_text().splitlines() if l.startswith("OK ")}

    to_fix = [t for t in broken if t not in fixed]
    print(f"[upload] {len(to_fix)} tasks to fix (skip {len(fixed)} done)")

    for i, tp in enumerate(to_fix, 1):
        print(f"[upload] ({i}/{len(to_fix)}) {tp}")
        if upload_task(args.token, local_root, tp, args.max_workers, log_dir):
            if verify_task_sample(api, args.token, tp):
                with open(progress_file, "a") as f:
                    f.write(f"OK {tp}\n")
                print(f"[ok] {tp}")
            else:
                with open(progress_file, "a") as f:
                    f.write(f"VERIFY_FAIL {tp}\n")
                print(f"[verify_fail] {tp}")
        else:
            with open(progress_file, "a") as f:
                f.write(f"FAIL {tp}\n")
            print(f"[fail] {tp}")

    print(f"[done] progress: {progress_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
