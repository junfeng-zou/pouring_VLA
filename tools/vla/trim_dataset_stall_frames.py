#!/usr/bin/env python3
"""
Copy VLA HDF5 episodes and drop leading / trailing frames where ||action_raw[:6]|| is below a threshold.

判定只用根部的 actions_raw（与 rgb 时间长度一致）；一旦确定要删的时间区间，会对**文件中所有与时间轴对齐的数据集**做同一索引切片（rgb、depth、state、actions、timestamps 等），保证每帧各类数据仍一一对齐。

Typical use (repo root):
  python tools/vla/trim_dataset_stall_frames.py \\
    --src data/vla_dataset --dst data/vla_dataset_processed --threshold 0.001
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np


def _list_episode_files(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))))
    return sorted(set(paths))


def _time_length_rgb(src: h5py.File) -> int:
    rgb = src["observations/images/rgb"]
    return int(rgb.shape[0])


def _trim_indices(actions_raw: np.ndarray, threshold: float) -> tuple[int, int] | None:
    """Return inclusive (start, end) indices to keep, or None if episode is empty after trim."""
    if actions_raw.ndim != 2 or actions_raw.shape[1] < 6:
        raise ValueError(f"actions_raw bad shape {actions_raw.shape}, need (T, >=6)")
    norms = np.linalg.norm(actions_raw[:, :6].astype(np.float64), axis=1)
    n = int(norms.shape[0])
    start = 0
    while start < n and float(norms[start]) < threshold:
        start += 1
    end = n - 1
    while end >= start and float(norms[end]) < threshold:
        end -= 1
    if start > end:
        return None
    return start, end


def _time_slice_for_dataset(
    src_ds: h5py.Dataset,
    sl: slice,
    t_old: int,
) -> tuple[np.ndarray, str]:
    """
    Return (array_to_store, mode) where mode is 'axis0', 'axis_last', or 'full'.

    - axis0: 第一维为时间 T，与 rgb 一致（当前 VLA 写入格式）
    - axis_last: 仅当第一维不是 T 且最后一维是 T 时，按最后一维切片（兼容时间维在末尾的布局）
    - full: 非按步展开的数组，原样拷贝
    """
    sh = src_ds.shape
    if not sh:
        return np.asarray(src_ds[()]), "full"
    if sh[0] == t_old:
        return np.asarray(src_ds[sl]), "axis0"
    if len(sh) >= 1 and sh[-1] == t_old and sh[0] != t_old:
        idx = (slice(None),) * (len(sh) - 1) + (sl,)
        return np.asarray(src_ds[idx]), "axis_last"
    return np.asarray(src_ds[()]), "full"


def _copy_dataset_sliced(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    sl: slice,
    t_old: int,
) -> None:
    data, _mode = _time_slice_for_dataset(src_ds, sl, t_old)
    kw: dict = {}
    if src_ds.compression:
        kw["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kw["compression_opts"] = src_ds.compression_opts
    d = dst_parent.create_dataset(name, data=data, **kw)
    for ak, av in src_ds.attrs.items():
        d.attrs[ak] = av


def _copy_group_sliced(
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    sl: slice,
    t_old: int,
) -> None:
    for ak, av in src_grp.attrs.items():
        dst_grp.attrs[ak] = av
    for key in src_grp.keys():
        item = src_grp[key]
        if isinstance(item, h5py.Dataset):
            _copy_dataset_sliced(item, dst_grp, key, sl, t_old)
        else:
            sub = dst_grp.create_group(key)
            _copy_group_sliced(item, sub, sl, t_old)


def _copy_root_attrs(src: h5py.File, dst: h5py.File, new_num_steps: int) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    dst.attrs["num_steps"] = new_num_steps


def process_episode(src_path: str, dst_path: str, threshold: float) -> tuple[bool, str]:
    with h5py.File(src_path, "r") as src:
        if "actions_raw" not in src:
            return False, "skip: no actions_raw"
        t_old = _time_length_rgb(src)
        ar = np.asarray(src["actions_raw"][:], dtype=np.float32)
        if ar.shape[0] != t_old:
            return False, f"skip: actions_raw T={ar.shape[0]} != rgb T={t_old}"
        bounds = _trim_indices(ar, threshold)
        if bounds is None:
            return False, "skip: all frames below threshold (empty)"
        start, end = bounds
        sl = slice(start, end + 1)
        new_t = end - start + 1

        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            for key in src.keys():
                item = src[key]
                if isinstance(item, h5py.Dataset):
                    _copy_dataset_sliced(item, dst, key, sl, t_old)
                else:
                    g = dst.create_group(key)
                    _copy_group_sliced(item, g, sl, t_old)
            _copy_root_attrs(src, dst, new_t)
            dst.attrs["trimmed_leading_trailing_stall"] = True
            dst.attrs["trim_stall_l2_threshold"] = threshold
            dst.attrs["trim_removed_leading"] = start
            dst.attrs["trim_removed_trailing"] = t_old - 1 - end

        with h5py.File(dst_path, "r") as verify:
            tr = int(verify["observations/images/rgb"].shape[0])
            if tr != new_t:
                return False, f"internal error: wrote rgb T={tr} expected {new_t}"
            for ds_path in ("actions", "actions_raw", "timestamps"):
                if ds_path in verify:
                    if int(verify[ds_path].shape[0]) != new_t:
                        return False, f"internal error: {ds_path} T mismatch after trim"

    return True, f"ok: {t_old} -> {new_t} frames (drop head {start}, tail {t_old - 1 - end})"


def copy_non_hdf5(src_dir: str, dst_dir: str) -> None:
    """Copy loose files (README, json, etc.); skip .hdf5 (handled separately)."""
    if not os.path.isdir(src_dir):
        return
    for name in os.listdir(src_dir):
        if name.endswith(".hdf5"):
            continue
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isfile(s):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/vla_dataset", help="Source folder with episode HDF5")
    p.add_argument("--dst", default="data/vla_dataset_processed", help="Output folder")
    p.add_argument("--threshold", type=float, default=0.001, help="L2 norm of action_raw[:6]")
    p.add_argument("--dry_run", action="store_true", help="Only print planned operations")
    args = p.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    if not os.path.isdir(src_dir):
        print(f"[错误] 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = _list_episode_files(src_dir)
    if not files:
        print(f"[错误] 未找到 episode_*.hdf5 或 *.hdf5: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] 将处理 {len(files)} 个 HDF5 -> {dst_dir}")
        for fp in files[:5]:
            print(f"  {fp}")
        if len(files) > 5:
            print(f"  ... 共 {len(files)}")
        sys.exit(0)

    os.makedirs(dst_dir, exist_ok=True)
    copy_non_hdf5(src_dir, dst_dir)

    ok_n = skip_n = 0
    for src_path in files:
        rel = os.path.relpath(src_path, src_dir)
        dst_path = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        ok, msg = process_episode(src_path, dst_path, args.threshold)
        print(f"{rel}: {msg}")
        if ok:
            ok_n += 1
        else:
            skip_n += 1

    print(f"\n完成: 写入 {ok_n} 个 episode，跳过 {skip_n} 个。输出目录: {dst_dir}")


if __name__ == "__main__":
    main()
