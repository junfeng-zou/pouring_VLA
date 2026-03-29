#!/usr/bin/env python3
"""统计 data_dir 下 HDF5 与 PouringHDF5VLADataset 一致的帧数（每个 actions 时间步 = 1 条样本）。"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_LORA_DIR = Path(__file__).resolve().parent
if str(_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(_LORA_DIR))

import h5py

from dataset_hdf5 import _decode_task, _list_hdf5_files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/vla_dataset")
    p.add_argument("--action_dim", type=int, default=7)
    args = p.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    files = _list_hdf5_files(data_dir)
    if not files:
        print(f"未找到 HDF5：{data_dir}（需要 episode_*.hdf5 或 *.hdf5）")
        raise SystemExit(1)

    total = 0
    print(f"目录: {data_dir}\n")
    for fp in files:
        with h5py.File(fp, "r") as f:
            actions = f["actions"]
            if actions.shape[-1] != args.action_dim:
                print(f"[SKIP] {fp}  actions 末维 {actions.shape[-1]} != {args.action_dim}")
                continue
            t = int(actions.shape[0])
            task = _decode_task(f.attrs.get("task", ""))
        rel = os.path.relpath(fp, data_dir)
        print(f"  {rel}: {t} 帧  (task={task!r})")
        total += t

    n_train_est = total
    print(f"\n合计: {total} 帧（= 数据集 __len__，与 train_lora 一条样本一帧一致）")
    val_ratio = 0.05
    if total > 10:
        n_val = max(1, int(total * val_ratio))
        n_train_est = total - n_val
    else:
        n_val = 1 if total > 0 else 0
        n_train_est = max(0, total - n_val)
    print(
        f"按 train_lora 默认 val_ratio={val_ratio}: 约 {n_train_est} 帧用于训练, {n_val} 帧用于验证"
    )


if __name__ == "__main__":
    main()
