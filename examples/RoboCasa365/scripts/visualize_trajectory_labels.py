#!/usr/bin/env python3
"""Overlay arm/base trajectory GT on dataset videos for sanity checks."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _video_path(lerobot_root: Path, chunk: int, camera: str, ep_name: str) -> Path:
    """LeRobot v2.1 stores videos under observation.images.{camera}."""
    return (
        lerobot_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{camera}"
        / f"{ep_name}.mp4"
    )


def _load_episode_df(traj_root: Path, episode_index: int) -> pd.DataFrame:
    chunk = episode_index // 1000
    ep_name = f"episode_{episode_index:06d}"
    parquet = traj_root / "data" / f"chunk-{chunk:03d}" / f"{ep_name}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    return pd.read_parquet(parquet)


def _uv_to_px(uv, w: int, h: int) -> tuple[int, int] | None:
    if uv is None or len(uv) != 2:
        return None
    u, v = float(uv[0]), float(uv[1])
    if u < 0 or v < 0:
        return None
    return int(round(u * w)), int(round(v * h))


def _blank_canvas(w: int = 256, h: int = 256) -> np.ndarray:
    """Image-plane canvas with grid when dataset videos are unavailable."""
    img = np.full((h, w, 3), 48, dtype=np.uint8)
    for t in np.linspace(0, 1, 5):
        x = int(round(t * (w - 1)))
        y = int(round(t * (h - 1)))
        cv2.line(img, (x, 0), (x, h - 1), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "no video (uv plane)",
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    return img


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    if not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _collect_past_trail(
    df: pd.DataFrame,
    camera: str,
    stream: str,
    frame_idx: int,
    trail_len: int,
    w: int,
    h: int,
) -> list[tuple[int, int]]:
    uv_col = f"trajectory.{stream}_uv.{camera}"
    vis_col = f"trajectory.{stream}_visible.{camera}"
    start = max(0, frame_idx - trail_len + 1)
    pts: list[tuple[int, int]] = []
    for i in range(start, frame_idx + 1):
        row = df.iloc[i]
        vis = row[vis_col]
        if isinstance(vis, (list, np.ndarray)):
            vis = bool(vis[0]) if len(vis) else False
        if not bool(vis):
            continue
        uv = row[uv_col]
        if isinstance(uv, (list, np.ndarray)):
            px = _uv_to_px(uv, w, h)
            if px is not None:
                pts.append(px)
    return pts


def _collect_anchored_future_trail(
    df: pd.DataFrame,
    camera: str,
    stream: str,
    frame_idx: int,
    w: int,
    h: int,
) -> list[tuple[int, int]]:
    """annotate_sim style: future world points projected with the anchor frame camera."""
    uv_col = f"trajectory.{stream}_future_uv.{camera}"
    vis_col = f"trajectory.{stream}_future_visible.{camera}"
    if uv_col not in df.columns:
        return []
    row = df.iloc[frame_idx]
    uvs = row[uv_col]
    vis = row[vis_col]
    pts: list[tuple[int, int]] = []
    if not isinstance(uvs, (list, np.ndarray)) or len(uvs) == 0:
        return pts
    for slot, uv in enumerate(uvs):
        slot_vis = True
        if isinstance(vis, (list, np.ndarray)) and slot < len(vis):
            slot_vis = bool(vis[slot])
        if not slot_vis:
            continue
        if isinstance(uv, (list, np.ndarray)) and len(uv) >= 2:
            px = _uv_to_px(uv[:2], w, h)
            if px is not None:
                pts.append(px)
    return pts


def _draw_trails(
    img: np.ndarray,
    df: pd.DataFrame,
    camera: str,
    frame_idx: int,
    trail_len: int,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    arm_pts = _collect_anchored_future_trail(df, camera, "arm", frame_idx, w, h)
    if not arm_pts:
        arm_pts = _collect_past_trail(df, camera, "arm", frame_idx, trail_len, w, h)
    base_pts = _collect_anchored_future_trail(df, camera, "base", frame_idx, w, h)
    if not base_pts:
        base_pts = _collect_past_trail(df, camera, "base", frame_idx, trail_len, w, h)

    for i in range(1, len(arm_pts)):
        cv2.line(out, arm_pts[i - 1], arm_pts[i], (0, 255, 80), 2, cv2.LINE_AA)
    for i in range(1, len(base_pts)):
        cv2.line(out, base_pts[i - 1], base_pts[i], (80, 180, 255), 2, cv2.LINE_AA)

    for px in arm_pts[:-1]:
        cv2.circle(out, px, 3, (0, 200, 60), -1, cv2.LINE_AA)
    for px in base_pts[:-1]:
        cv2.circle(out, px, 3, (60, 140, 255), -1, cv2.LINE_AA)
    if arm_pts:
        cv2.circle(out, arm_pts[-1], 6, (0, 255, 0), -1, cv2.LINE_AA)
    if base_pts:
        cv2.circle(out, base_pts[-1], 6, (255, 120, 0), -1, cv2.LINE_AA)

    return out


def _render_frame_grid(
    df: pd.DataFrame,
    lerobot_root: Path,
    episode_index: int,
    frame_idx: int,
    trail_len: int,
    caps: dict[str, cv2.VideoCapture] | None = None,
) -> np.ndarray:
    chunk = episode_index // 1000
    ep_name = f"episode_{episode_index:06d}"
    n_frames = len(df)
    panels = []
    for camera in CAMERAS:
        frame = None
        if caps is not None and camera in caps:
            ok, bgr = caps[camera].read()
            if ok:
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame is None:
            frame = _read_frame(_video_path(lerobot_root, chunk, camera, ep_name), frame_idx)
        if frame is None:
            frame = _blank_canvas()
        frame = _draw_trails(frame, df, camera, frame_idx, trail_len)
        cv2.putText(
            frame,
            camera.replace("robot0_", ""),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            camera.replace("robot0_", ""),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(frame)

    grid = np.concatenate(panels, axis=1)
    legend_h = 36
    legend = np.full((legend_h, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        legend,
        f"ep{episode_index} frame {frame_idx}/{n_frames - 1}  "
        "green=arm(EEF)  orange=base  future trail (annotate_sim)",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([legend, grid], axis=0)


def _transcode_h264(src: Path, dst: Path) -> Path:
    """Re-mux to H.264/avc1 so browsers and IDE preview can play the mp4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return src
    tmp = dst.with_suffix(".h264.tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(dst)
    if src != dst and src.exists():
        src.unlink()
    return dst


def export_overlay_mp4(
    lerobot_root: Path,
    traj_root: Path,
    episode_index: int,
    output_dir: Path,
    trail_len: int,
    fps: float = 10.0,
    frame_step: int = 2,
) -> Path:
    chunk = episode_index // 1000
    ep_name = f"episode_{episode_index:06d}"
    df = _load_episode_df(traj_root, episode_index)
    n_frames = len(df)

    caps: dict[str, cv2.VideoCapture] = {}
    for camera in CAMERAS:
        path = _video_path(lerobot_root, chunk, camera, ep_name)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        caps[camera] = cap

    out_path = output_dir / f"ep{episode_index:03d}_overlay.mp4"
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = None
    try:
        for frame_idx in range(n_frames):
            panels = []
            for camera in CAMERAS:
                ok, bgr = caps[camera].read()
                if not ok:
                    raise RuntimeError(
                        f"Video ended early at frame {frame_idx} for {camera}"
                    )
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = _draw_trails(frame, df, camera, frame_idx, trail_len)
                cv2.putText(
                    frame,
                    camera.replace("robot0_", ""),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    camera.replace("robot0_", ""),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(frame)

            if frame_idx % max(1, frame_step) != 0:
                continue

            grid = np.concatenate(panels, axis=1)
            legend_h = 36
            legend = np.full((legend_h, grid.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(
                legend,
                f"ep{episode_index} frame {frame_idx}/{n_frames - 1}  "
                "green=arm(EEF)  orange=base  future trail (annotate_sim)",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            bgr = cv2.cvtColor(np.concatenate([legend, grid], axis=0), cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
            writer.write(bgr)
    finally:
        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()
    if raw_path.exists():
        return _transcode_h264(raw_path, out_path)
    return out_path


def visualize_episode(
    lerobot_root: Path,
    traj_root: Path,
    episode_index: int,
    output_dir: Path,
    frame_indices: list[int],
    trail_len: int = 40,
) -> list[Path]:
    chunk = episode_index // 1000
    df = _load_episode_df(traj_root, episode_index)
    n_frames = len(df)
    saved: list[Path] = []

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= n_frames:
            continue
        out_img = _render_frame_grid(
            df, lerobot_root, episode_index, frame_idx, trail_len
        )
        out_path = output_dir / f"ep{episode_index:03d}_frame{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    return saved


def plot_full_episode(
    traj_root: Path,
    episode_index: int,
    output_dir: Path,
) -> Path:
    """Matplotlib overview: full-episode uv paths per camera."""
    import matplotlib.pyplot as plt

    chunk = episode_index // 1000
    ep_name = f"episode_{episode_index:06d}"
    df = pd.read_parquet(
        traj_root / "data" / f"chunk-{chunk:03d}" / f"{ep_name}.parquet"
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(
        f"PickPlace ep{episode_index} — full trajectory GT (normalized uv)",
        fontsize=12,
    )
    for j, camera in enumerate(CAMERAS):
        for i, stream in enumerate(("arm", "base")):
            ax = axes[i, j]
            uv_col = f"trajectory.{stream}_uv.{camera}"
            vis_col = f"trajectory.{stream}_visible.{camera}"
            uvs = []
            for _, row in df.iterrows():
                vis = row[vis_col]
                if isinstance(vis, (list, np.ndarray)):
                    vis = bool(vis[0]) if len(vis) else False
                if not bool(vis):
                    continue
                uv = row[uv_col]
                if isinstance(uv, (list, np.ndarray)) and float(uv[0]) >= 0:
                    uvs.append([float(uv[0]), float(uv[1])])
            uvs = np.array(uvs) if uvs else np.zeros((0, 2))
            color = "tab:green" if stream == "arm" else "tab:orange"
            if len(uvs):
                ax.plot(uvs[:, 0], 1.0 - uvs[:, 1], color=color, lw=1.2, alpha=0.9)
                ax.scatter(uvs[0, 0], 1.0 - uvs[0, 1], c="blue", s=20, label="start")
                ax.scatter(uvs[-1, 0], 1.0 - uvs[-1, 1], c="red", s=20, label="end")
            ax.set_title(camera.replace("robot0_", ""), fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
            if j == 0:
                ax.set_ylabel(f"{stream} v (image down)")
            if i == 1:
                ax.set_xlabel("u")
    fig.tight_layout()
    out_path = output_dir / f"ep{episode_index:03d}_full_trajectory.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="Original lerobot dir (videos)",
    )
    parser.add_argument(
        "--traj-root",
        type=Path,
        required=True,
        help="lerobot_traj dir (parquet with trajectory.* columns)",
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Frame indices; default picks quartiles of episode length",
    )
    parser.add_argument("--trail-len", type=int, default=40)
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="Export full-episode overlay mp4 (3 views side-by-side)",
    )
    parser.add_argument("--mp4-fps", type=float, default=10.0)
    parser.add_argument(
        "--mp4-step",
        type=int,
        default=2,
        help="Write every Nth frame into mp4 (default 2 → ~10fps from 20fps data)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        parquet = args.traj_root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        n = len(pd.read_parquet(parquet))
        if args.frames is None:
            frames = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        else:
            frames = args.frames
        paths = visualize_episode(
            args.lerobot_root,
            args.traj_root,
            ep,
            args.output_dir,
            frames,
            trail_len=args.trail_len,
        )
        for p in paths:
            print(p)
        if args.export_mp4:
            mp4 = export_overlay_mp4(
                args.lerobot_root,
                args.traj_root,
                ep,
                args.output_dir,
                trail_len=args.trail_len,
                fps=args.mp4_fps,
                frame_step=args.mp4_step,
            )
            print(mp4)
        full = plot_full_episode(args.traj_root, ep, args.output_dir)
        print(full)


if __name__ == "__main__":
    main()
