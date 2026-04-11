# `scripts/` 说明

按**功能**分子目录；是否走 Isaac Lab 在表格里单独标明（`isaaclab.sh -p`）。

## `scripts/collect/` — 数据采集

| 脚本 | 用途 | 运行方式 |
|------|------|------|
| `teleop_collect.py` | 仿真里手柄采集 + HDF5 | Isaac Lab |
| `auto_collect.py` | **自动采集（无需手柄）**，基于预设规则执行倒水动作 | Isaac Lab |
| `real_teleop_collect.py` | 真机采集（拖拽 / 手柄） | 仓库根目录普通 `python` |

## `scripts/env/` — 环境自检

| 脚本 | 用途 | 运行方式 |
|------|------|----------|
| `run_pouring_env.py` | PouringEnv 冒烟测试 | Isaac Lab |

## `scripts/inference/` — 策略推理与联机

| 脚本 | 用途 | 运行方式 |
|------|------|----------|
| `pouring_sim_policy_server.py` | 仿真侧 TCP 桥接，接远程策略 | Isaac Lab |
| `openvla_lora_pouring_infer.py` | 单进程：仿真 + 本机 OpenVLA+LoRA | Isaac Lab |
| `openvla_lora_policy_client.py` | 仅 GPU 推理，与上面对 `pouring_sim_policy_server` 配对 | 普通 `python`（建议独立 venv） |

## `scripts/robot/` — 机械臂资源与资产

| 脚本 | 用途 | 运行方式 |
|------|------|----------|
| `convert_dobot_urdf.py` | URDF → USD | Isaac Lab / isaac-sim `python.sh` |
| `verify_dobot_nova5.py` | Nova5 USD 验证 | Isaac Lab |
| `dobot_nova5_cfg.py` | `ArticulationCfg` 片段（供其它脚本 import / 参考） | 非入口，按需 import |

## `scripts/robot/rviz_replay/` — HDF5 关节轨迹在 RViz2 中回放

| 内容 | 说明 |
|------|------|
| `launch/rviz_nova5_replay.launch.py` | 静态 TF、`/robot_description`、`robot_state_publisher`、可选 RViz |
| `hdf5_rviz_replay.py` | 读 HDF5，发布 `/joint_states` |
| `robot_description_topic_pub.py` | 将 URDF 字符串发到 `/robot_description` 话题 |
| `urdf/`、`meshes/` | Dobot 各机型 URDF 与 STL（供 RViz 网格） |

示例（仓库根目录、已 `source` ROS2）：

```bash
ros2 launch scripts/robot/rviz_replay/launch/rviz_nova5_replay.launch.py
python3 scripts/robot/rviz_replay/hdf5_rviz_replay.py --episode <episode.hdf5> --mode joints
```

运行示例：

```bash
# 传给用户脚本的参数必须写在 `--` 之后，否则会被 Isaac Kit 吞掉
isaaclab.sh -p scripts/collect/teleop_collect.py -- --headless
isaaclab.sh -p scripts/inference/pouring_sim_policy_server.py -- --port 9875

python scripts/inference/openvla_lora_policy_client.py --host 127.0.0.1 --port 9875 ...
```

退出仿真时若出现 **段错误 / omni.graph 崩溃栈**，多为 Kit 在异常退出时的已知清理问题；尽量只按一次 Ctrl+C 等待进程结束，或先连上客户端再正常退出。
