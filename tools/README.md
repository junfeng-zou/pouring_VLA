# `tools/` 说明

库与命令行小工具按主题分子目录（均为 `python -m` 或 `python tools/...` 从仓库根运行）。

## `tools/vla/`

OpenVLA 推理与数据集写入：

- `vla_data_writer.py` — HDF5 episode 写入（采集脚本依赖）
- `openvla_lora_runtime.py` — 加载 LoRA、Vicuna 提示、反归一化、单步推理
- `pouring_policy_wire.py` — 仿真↔策略 TCP 帧协议

## `tools/teleop/`

- `gamepad_receiver.py` — 仿真端 UDP 收手柄
- `gamepad_sender.py` — 本地 PC 发手柄

## `tools/export/`

- `export_readable_traj.py` — 单条 episode → CSV/图
- `export_episode_readable_all.py` — 单条 episode → 可读目录（PNG/NPY 等）

## `tools/dev/`

- `cam_quat_helper.py` — 相机四元数 UI ↔ `CameraCfg` 换算

导入示例：

```python
from tools.teleop.gamepad_receiver import GamepadReceiver
from tools.vla.vla_data_writer import EpisodeRecorder
```
