#!/usr/bin/env python3
"""
Download RoboCasa365 kitchen assets from HuggingFace (hf-mirror) and extract
into the same paths as ``robocasa/scripts/download_kitchen_assets.py``.

Default repo: twilighted/Robocasa365-Assets

Because HF upload names vary, this script:
  1. Snapshots the whole repo (recommended), then auto-matches ``*.zip`` by keyword.
  2. Or downloads a single zip when ``--zip-map key=filename.zip`` is given.

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python download_kitchen_assets_hf_mirror.py --snapshot
  python download_kitchen_assets_hf_mirror.py --list-zips   # after snapshot
  python download_kitchen_assets_hf_mirror.py --types fixtures_lw objs_lw --force
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ASSET_MAP: dict[str, dict[str, object]] = {
    "tex": {
        "folder_name": "textures",
        "hf_zips": ("textures.zip",),
    },
    "tex_generative": {
        "folder_name": "generative_textures",
        "hf_zips": ("generative_textures.zip",),
    },
    "fixtures_lw": {
        "folder_name": "fixtures",
        "hf_zips": ("fixtures.zip", "fixtures_lightwheel.zip"),
    },
    "objs_objaverse": {
        "folder_name": "objects/objaverse",
        "hf_zips": ("objaverse.zip",),
    },
    "objs_aigen": {
        "folder_name": "objects/aigen_objs",
        "hf_zips": ("aigen_objs.zip",),
    },
    "objs_lw": {
        "folder_name": "objects/lightwheel",
        "hf_zips": ("lightwheel.zip", "objects_lightwheel.zip"),
    },
}

DEFAULT_REPO = "twilighted/Robocasa365-Assets"

COMPLETENESS_CHECKS: dict[str, tuple[Path, ...]] = {
    "tex": (Path("textures"),),
    "tex_generative": (Path("generative_textures"),),
    "fixtures_lw": (Path("fixtures/windows/Window069/model.xml"),),
    "objs_objaverse": (Path("objects/objaverse"),),
    "objs_aigen": (Path("objects/aigen_objs"),),
    "objs_lw": (Path("objects/lightwheel"),),
}


def _robocasa_assets_root() -> Path:
    import robocasa

    return Path(robocasa.__path__[0]) / "models" / "assets"


def _mirror_base() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return endpoint


def _classify_zip_name(name: str) -> str | None:
    """Map an arbitrary zip filename to an ASSET_MAP key."""
    n = name.lower()
    if re.search(r"generative.*texture|texture.*generative", n):
        return "tex_generative"
    if "fixture" in n and "lightwheel" in n:
        return "fixtures_lw"
    if n in ("fixtures.zip", "fixtures_lightwheel.zip") or (
        "fixture" in n and n.endswith(".zip")
    ):
        return "fixtures_lw"
    if "objaverse" in n:
        return "objs_objaverse"
    if "aigen" in n:
        return "objs_aigen"
    if "lightwheel" in n and "object" in n:
        return "objs_lw"
    if n in ("lightwheel.zip", "objects_lightwheel.zip"):
        return "objs_lw"
    if "lightwheel" in n and "fixture" not in n:
        return "objs_lw"
    if "generative" in n:
        return "tex_generative"
    if "texture" in n and "generative" not in n:
        return "tex"
    return None


def _discover_zips(search_root: Path) -> dict[str, Path]:
    """Find best zip per asset key under snapshot or cache."""
    found: dict[str, Path] = {}
    for zp in sorted(search_root.rglob("*.zip")):
        if zp.stat().st_size < 1024:
            continue
        key = _classify_zip_name(zp.name)
        if key is None:
            continue
        prev = found.get(key)
        if prev is None or zp.stat().st_size > prev.stat().st_size:
            found[key] = zp
    return found


def _snapshot_download(repo_id: str, dest: Path) -> Path:
    """Download full HF repo; try model then dataset repo_type."""
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    for cli in ("hf", "huggingface-cli"):
        if shutil.which(cli) is None:
            continue
        for repo_type in ("model", "dataset"):
            cmd = [
                cli,
                "download",
                repo_id,
                "--repo-type",
                repo_type,
                "--local-dir",
                str(dest),
            ]
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, env=env)
                return dest
            except subprocess.CalledProcessError as e:
                print(f"{cli} ({repo_type}) failed: {e}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Install huggingface_hub: pip install huggingface_hub  (or use `hf download`)"
        ) from e

    last_err: Exception | None = None
    for repo_type in ("model", "dataset"):
        try:
            print(f"snapshot_download(repo_type={repo_type}) -> {dest}")
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            return dest
        except Exception as e:
            last_err = e
            print(f"snapshot_download ({repo_type}) failed: {e}")
    raise RuntimeError(f"Could not snapshot {repo_id}") from last_err


def _resolve_url(repo_id: str, filename: str, repo_type: str) -> str:
    base = _mirror_base()
    if repo_type == "dataset":
        return f"{base}/datasets/{repo_id}/resolve/main/{filename}"
    return f"{base}/{repo_id}/resolve/main/{filename}"


def _download_single_zip(repo_id: str, filename: str, dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    for cli in ("hf", "huggingface-cli"):
        if shutil.which(cli) is None:
            continue
        for repo_type in ("model", "dataset"):
            cmd = [
                cli,
                "download",
                repo_id,
                filename,
                "--repo-type",
                repo_type,
                "--local-dir",
                str(dest.parent),
            ]
            try:
                subprocess.run(cmd, check=True, env=os.environ.copy())
                out = dest.parent / filename
                if out.is_file():
                    return out
            except subprocess.CalledProcessError:
                continue

    import urllib.request

    for repo_type in ("model", "dataset"):
        url = _resolve_url(repo_id, filename, repo_type)
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            print(f"  failed ({repo_type}): {e}")
    return None


def _is_bundle_complete(asset_key: str, assets_root: Path) -> bool:
    for rel in COMPLETENESS_CHECKS.get(asset_key, ()):
        p = assets_root / rel
        if rel.suffix == ".xml":
            if not p.is_file():
                return False
        elif not p.is_dir() or not any(p.iterdir()):
            return False
    return bool(COMPLETENESS_CHECKS.get(asset_key))


def _fix_extract_layout(assets_root: Path, target_folder: Path) -> None:
    target_folder = target_folder.resolve()
    assets_root = assets_root.resolve()

    nested = target_folder / target_folder.name
    if nested.is_dir():
        print(f"Merging nested {nested} -> {target_folder}")
        for child in nested.iterdir():
            dest = target_folder / child.name
            if dest.exists() and dest.is_dir():
                shutil.copytree(child, dest, dirs_exist_ok=True)
            elif not dest.exists():
                shutil.move(str(child), str(dest))
        shutil.rmtree(nested)

    if target_folder.name == "fixtures":
        for name in ("windows", "stoves", "sinks", "microwaves", "fridges", "dishwashers"):
            misplaced = assets_root / name
            dest = target_folder / name
            if misplaced.is_dir() and not dest.exists():
                shutil.move(str(misplaced), str(dest))


def _extract_zip(
    zip_path: Path,
    target_folder: Path,
    assets_root: Path,
    asset_key: str,
    *,
    force: bool,
) -> None:
    target_folder = target_folder.resolve()
    download_dir = target_folder.parent
    download_dir.mkdir(parents=True, exist_ok=True)

    if _is_bundle_complete(asset_key, assets_root) and not force:
        print(f"Skip extract (already complete): {asset_key}")
        return

    staged = download_dir / f"{target_folder.name}.zip"
    if zip_path.resolve() != staged.resolve():
        shutil.copy2(zip_path, staged)

    print(f"Extracting {staged} -> {download_dir}")
    with ZipFile(staged, "r") as zf:
        zf.extractall(path=download_dir)
    if staged.is_file():
        staged.unlink()
    _fix_extract_layout(assets_root, target_folder)


def _validate(assets_root: Path) -> list[str]:
    missing: list[str] = []
    for key, rels in COMPLETENESS_CHECKS.items():
        for rel in rels:
            p = assets_root / rel
            ok = p.is_file() if rel.suffix == ".xml" else p.is_dir() and any(p.iterdir())
            if not ok:
                missing.append(f"{key}: {rel}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_ASSETS_REPO", DEFAULT_REPO))
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(ASSET_MAP.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Download entire HF repo first (recommended when per-file URLs 404)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Where to store HF snapshot (default: cache/repo_snapshot)",
    )
    parser.add_argument(
        "--list-zips",
        action="store_true",
        help="List discovered zips under cache/snapshot and exit",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}  repo={args.repo}")

    assets_root = _robocasa_assets_root()
    cache_dir = args.cache_dir or (assets_root / ".hf_download_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.snapshot_dir or (cache_dir / "repo_snapshot")

    if args.snapshot and not args.skip_download:
        _snapshot_download(args.repo, snapshot_dir)

    search_roots = [snapshot_dir, cache_dir] if snapshot_dir.exists() else [cache_dir]
    zip_by_key = {}
    for root in search_roots:
        zip_by_key.update(_discover_zips(root))

    if args.list_zips:
        if not zip_by_key:
            print(f"No zips under {search_roots}. Run with --snapshot first.")
            for root in search_roots:
                if root.exists():
                    for z in sorted(root.rglob("*.zip")):
                        print(f"  unclassified: {z}")
            return 1
        print("Discovered zips:")
        for k, p in sorted(zip_by_key.items()):
            print(f"  {k}: {p} ({p.stat().st_size / 1e9:.2f} GB)")
        return 0

    types = list(ASSET_MAP.keys()) if "all" in args.types else args.types

    for key in types:
        target = assets_root / str(ASSET_MAP[key]["folder_name"])
        zip_path = zip_by_key.get(key)

        if zip_path is None and not args.skip_download:
            for zname in ASSET_MAP[key]["hf_zips"]:
                cand = cache_dir / str(zname)
                zip_path = _download_single_zip(args.repo, str(zname), cand)
                if zip_path is not None:
                    break

        if zip_path is None:
            print(f"[{key}] no zip found; skipped.", file=sys.stderr)
            continue

        _extract_zip(zip_path, target, assets_root, key, force=args.force)

    missing = _validate(assets_root)
    if missing:
        print("\nStill missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "\nTry:\n"
            f"  export HF_ENDPOINT=https://hf-mirror.com\n"
            f"  python {Path(__file__).name} --snapshot --list-zips\n"
            f"  python {Path(__file__).name} --types fixtures_lw objs_lw tex --force\n"
            "If repo is private: hf auth login\n"
            "If zips use other names, place them under .hf_download_cache/ and re-run.",
            file=sys.stderr,
        )
        return 1

    print("All required kitchen asset bundles look complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
