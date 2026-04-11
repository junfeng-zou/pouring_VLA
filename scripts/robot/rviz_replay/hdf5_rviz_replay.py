#!/usr/bin/env python3
"""
在 RViz2 中回放 HDF5 中的机械臂关节轨迹或「积分 actions」轨迹（默认 Nova5 URDF，可用 --urdf 换 CR5 等）。

依赖：已 source ROS2（如 Humble），且安装 rclpy、sensor_msgs、std_msgs、h5py、numpy。

真机数据（real_teleop_server）：
  - observations/state 为 19 维：关节角(度)、速度、夹爪、末端位姿 xyz(m)+rpy(°)。
  - actions[t] 表示从 state[t-1] 到 state[t] 的末端增量：0:3 为位置差(米)，3:6 为姿态增量(弧度，等价度×π/180)；
    积分重建时 pose[t]=pose[t-1]+decode(actions[t])，故 actions[0] 不参与积分（与首帧手柄量无关）。

用法示例
--------
  # 终端1：map→dummy_link 静态 TF + /robot_description 话题 + robot_state_publisher + RViz
  source /opt/ros/humble/setup.bash
  ros2 launch scripts/robot/rviz_replay/launch/rviz_nova5_replay.launch.py

  # 终端2：按 state 中的关节角回放（最贴近真机姿态）
  python3 scripts/robot/rviz_replay/hdf5_rviz_replay.py \\
      --episode data/vla_dataset_real/episode_0000.hdf5 --mode joints

  # 用 actions 积分末端位姿 + 数值 IK 驱动模型（对照动作是否合理）
  python3 scripts/robot/rviz_replay/hdf5_rviz_replay.py \\
      --episode data/vla_dataset_real/episode_0000.hdf5 --mode integrate_actions

若 STL 网格缺失，可在 RViz 中仍可通过 TF / 简化模型查看运动；可用 --mesh_dir 将
package://dobot_rviz/meshes/ 替换为本地 meshes 目录（其下应有 nova5/ 或 cr5/*.STL）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.export.recompute_actions_raw_from_state_fk import (  # noqa: E402
    _rot_to_rotvec,
    _rpy_to_rot,
    fk_link6_from_joint6,
    load_joint_chain,
)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
except ImportError as e:
    rclpy = None
    Node = object  # type: ignore[misc, assignment]
    _ROS_IMPORT_ERROR = e
else:
    _ROS_IMPORT_ERROR = None

_DEFAULT_PKG = Path(__file__).resolve().parent
DEFAULT_URDF = _DEFAULT_PKG / "urdf" / "nova5_robot.urdf"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def infer_joint_rad(row6: np.ndarray, unit: str) -> np.ndarray:
    j = np.asarray(row6, dtype=np.float64).reshape(6)
    if unit == "deg":
        return np.deg2rad(j)
    if unit == "rad":
        return j
    # auto：真机常为几十度；仿真 Isaac 关节多在 ±π
    if np.max(np.abs(j)) > 6.5:
        return np.deg2rad(j)
    return j


def integrate_cartesian_from_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """返回 (T,6) 末端位姿：xyz(m), rx,ry,rz(度)。

    与 real_teleop_server 一致：HDF5 第 t 行的 action[t] 对应 state[t]-state[t-1]（t>=1），
    故 pose[t] = pose[t-1] + decode(action[t])；actions[0] 不使用。
    """
    if state.shape[1] < 19:
        raise ValueError(
            f"integrate_actions 需要 state 至少 19 维（含末端 xyz+rpy），当前 {state.shape[1]}"
        )
    T = state.shape[0]
    out = np.zeros((T, 6), dtype=np.float64)
    out[0] = state[0, 13:19].astype(np.float64)
    for t in range(1, T):
        a = actions[t].astype(np.float64)
        prev = out[t - 1].copy()
        prev[0:3] = prev[0:3] + a[0:3]
        prev[3:6] = prev[3:6] + a[3:6] * (180.0 / np.pi)
        out[t] = prev
    return out


def ik_solve_step(
    chain,
    p_tgt: np.ndarray,
    R_tgt: np.ndarray,
    q_init: np.ndarray,
    *,
    iters: int = 80,
    lam: float = 0.05,
    rot_weight: float = 0.35,
) -> np.ndarray:
    """简单阻尼最小二乘 + 数值雅可比，求 6 轴角(rad)。"""
    q = np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(iters):
        p_fk, R_fk = fk_link6_from_joint6(chain, q)
        e_p = p_tgt - p_fk
        e_r = _rot_to_rotvec(R_fk.T @ R_tgt)
        e = np.concatenate([e_p, e_r * rot_weight])
        if np.linalg.norm(e_p) < 1e-5 and np.linalg.norm(e_r) < 1e-4:
            break
        J = np.zeros((6, 6), dtype=np.float64)
        eps = 1e-4
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            pi, Ri = fk_link6_from_joint6(chain, q + dq)
            J[:3, i] = (pi - p_fk) / eps
            J[3:, i] = (_rot_to_rotvec(R_fk.T @ Ri)) / eps
        JTJ = J.T @ J + lam * np.eye(6)
        dq_step = np.linalg.solve(JTJ, J.T @ e)
        q = q + dq_step
        q = np.clip(q, -6.5, 6.5)
    return q


def load_hdf5_trajectory(
    episode: Path,
    mode: str,
    joint_unit: str,
    chain,
) -> tuple[np.ndarray, str]:
    """返回 (T,6) 关节弧度矩阵与模式描述。"""
    with h5py.File(episode, "r") as f:
        state = np.asarray(f["observations/state"][:], dtype=np.float32)
        actions = np.asarray(f["actions"][:], dtype=np.float32)
    if actions.shape[0] != state.shape[0]:
        raise ValueError(
            f"state 与 actions 步数不一致: {state.shape[0]} vs {actions.shape[0]}"
        )

    T = state.shape[0]
    q_seq = np.zeros((T, 6), dtype=np.float64)

    if mode == "joints":
        for t in range(T):
            q_seq[t] = infer_joint_rad(state[t, :6], joint_unit)
        return q_seq, "关节角来自 observations/state[:,0:6]"

    if mode == "integrate_actions":
        cart = integrate_cartesian_from_actions(state, actions)
        q_prev = infer_joint_rad(state[0, :6], joint_unit)
        q_seq[0] = q_prev
        for t in range(1, T):
            rx, ry, rz = cart[t, 3:6]
            R_tgt = _rpy_to_rot(np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz))
            p_tgt = cart[t, 0:3]
            q_prev = ik_solve_step(chain, p_tgt, R_tgt, q_prev)
            q_seq[t] = q_prev
        return q_seq, "末端位姿由 actions 积分 + URDF 数值 IK（与 Dobot 坐标系可能有偏差，仅供肉眼对照）"

    raise ValueError(f"未知 mode: {mode}")


class Hdf5JointReplayNode(Node):
    def __init__(
        self,
        q_seq: np.ndarray,
        rate_hz: float,
        loop: bool,
        topic: str,
    ):
        super().__init__("hdf5_rviz_replay")
        self._q = q_seq
        self._n = int(q_seq.shape[0])
        self._idx = 0
        self._loop = loop
        self._pub = self.create_publisher(JointState, topic, 10)
        period = 1.0 / max(rate_hz, 0.1)
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"回放 {self._n} 步 @ {rate_hz:.1f} Hz, topic={topic!r}, loop={loop}"
        )

    def _tick(self) -> None:
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(x) for x in self._q[self._idx]]
        self._pub.publish(msg)
        self._idx += 1
        if self._idx >= self._n:
            if self._loop:
                self._idx = 0
                self.get_logger().info("循环：回到第 0 步")
            else:
                self.get_logger().info("回放结束，保持最后一帧")
                self._idx = self._n - 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HDF5 → RViz2 JointState 回放（默认 Nova5）")
    p.add_argument("--episode", type=str, required=True, help="episode_*.hdf5 路径")
    p.add_argument(
        "--mode",
        type=str,
        choices=["joints", "integrate_actions"],
        default="joints",
        help="joints=直接用 state 关节角；integrate_actions=积分 actions 再 IK",
    )
    p.add_argument(
        "--joint_unit",
        type=str,
        choices=["auto", "deg", "rad"],
        default="auto",
        help="state 前 6 维单位（仿真 HDF5 多为 rad，真机为度）",
    )
    p.add_argument(
        "--urdf",
        type=str,
        default=str(DEFAULT_URDF),
        help="机器人 URDF 路径（默认 nova5_robot.urdf，可改为 cr5_robot.urdf）",
    )
    p.add_argument("--base_link", type=str, default="base_link")
    p.add_argument("--ee_link", type=str, default="Link6")
    p.add_argument("--rate_hz", type=float, default=10.0, help="发布频率（与采集 record_hz 接近较直观）")
    p.add_argument("--loop", action="store_true", help="结束后从头循环")
    p.add_argument("--topic", type=str, default="/joint_states", help="JointState 话题")
    p.add_argument(
        "--compare_integrate",
        action="store_true",
        help="integrate_actions 模式下打印积分末端与 state 中末端的 RMS 位置误差(m)",
    )
    return p.parse_args()


def main() -> int:
    if _ROS_IMPORT_ERROR is not None:
        print(
            "无法导入 rclpy。请先安装 ROS2 并在当前 shell 执行：\n"
            "  source /opt/ros/<distro>/setup.bash\n"
            f"原始错误: {_ROS_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 1

    args = parse_args()
    episode = Path(args.episode).expanduser()
    if not episode.is_file():
        print(f"找不到 HDF5: {episode}", file=sys.stderr)
        return 1

    urdf_path = Path(args.urdf).expanduser()
    if not urdf_path.is_file():
        print(f"找不到 URDF: {urdf_path}", file=sys.stderr)
        return 1

    chain = load_joint_chain(urdf_path, args.base_link, args.ee_link)

    if args.compare_integrate and args.mode == "integrate_actions":
        with h5py.File(episode, "r") as f:
            state = np.asarray(f["observations/state"][:], dtype=np.float32)
            actions = np.asarray(f["actions"][:], dtype=np.float32)
        cart_int = integrate_cartesian_from_actions(state, actions)
        if state.shape[1] >= 19:
            ref = state[:, 13:16].astype(np.float64)
            err = np.linalg.norm(cart_int[:, :3] - ref, axis=1)
            print(
                f"[compare_integrate] pos RMSE={float(np.sqrt(np.mean(err**2))):.6f} m, "
                f"max={float(np.max(err)):.6f} m（若很大，说明 Dobot 与 URDF 坐标系不一致或首帧需跳过）"
            )

    q_seq, desc = load_hdf5_trajectory(episode, args.mode, args.joint_unit, chain)
    print(f"[INFO] {episode.name}: {desc}")
    print(f"[INFO] 步数={q_seq.shape[0]}, 发布 /joint_states @ {args.rate_hz} Hz")

    rclpy.init()
    try:
        node = Hdf5JointReplayNode(
            q_seq,
            rate_hz=args.rate_hz,
            loop=args.loop,
            topic=args.topic,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
