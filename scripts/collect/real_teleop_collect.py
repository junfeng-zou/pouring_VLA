#!/usr/bin/env python3
"""
DOBOT CR5 — 真机 VLA 数据采集
=================================

支持两种采集模式：
  1. 拖拽模式 (drag)  — 操作员手动拖动机器人，推荐
  2. 手柄模式 (gamepad) — 手柄遥操作，通过 MovL 笛卡尔运动

采集数据格式与仿真脚本 (teleop_collect.py) 一致，输出 HDF5 文件用于 OpenVLA-7B 微调。

启动：
    python scripts/collect/real_teleop_collect.py --mode drag
    python scripts/collect/real_teleop_collect.py --mode gamepad --port 9876

手柄控制映射（Xbox / 通用双摇杆）：
    左摇杆 X        → EE Y  (左/右)
    左摇杆 Y        → EE X  (前/后)
    右摇杆 X        → EE yaw 旋转
    右摇杆 Y        → EE Z  (上/下)
    LT   (轴 4)     → pitch-
    RT   (轴 5)     → pitch+
    LB   (按钮 4)   → roll-
    RB   (按钮 5)   → roll+
    A    (按钮 0)   → 切换录制
    B    (按钮 1)   → 丢弃当前 episode
    X    (按钮 2)   → 切换夹爪
    Y    (按钮 3)   → 复位（回初始位姿）
    Start(按钮 7)   → 退出

拖拽模式下按键控制（通过手柄或键盘）：
    A / 键盘 'r'    → 切换录制
    B / 键盘 'd'    → 丢弃当前 episode
    X / 键盘 'g'    → 切换夹爪
    Start / ESC     → 退出
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

# -- 让项目根目录可导入 --
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from dobot_cr5 import DobotCR5
from tools.teleop.gamepad_receiver import GamepadReceiver
from tools.vla.vla_data_writer import EpisodeRecorder, next_episode_path


# ═══════════════════════════════════════════════════════════════════════════
# 夹爪控制器（接口预留，用户后续通过 GPIO 补充实现）
# ═══════════════════════════════════════════════════════════════════════════

class GripperController:
    """夹爪控制器 — GPIO 控制接口（待用户实现）

    用法示例：
        gripper = GripperController()
        gripper.open()
        gripper.close()
        state = gripper.get_state()  # 0.0=关闭, 1.0=打开（支持中间过渡状态）
    """

    def __init__(self):
        self._target_state = 1.0  # 1.0 为完全打开
        self._start_state = 1.0
        self._last_toggle_time = 0.0
        self._actuation_time = 0.5  # 物理夹爪开合所需的时间（秒）

    def open(self):
        """打开夹爪 — TODO: 用户添加 GPIO 控制代码"""
        if self._target_state != 1.0:
            self._start_state = self.get_state()
            self._target_state = 1.0
            self._last_toggle_time = time.time()
            print("[GRIPPER] 打开指令下发..")

    def close(self):
        """关闭夹爪 — TODO: 用户添加 GPIO 控制代码"""
        if self._target_state != 0.0:
            self._start_state = self.get_state()
            self._target_state = 0.0
            self._last_toggle_time = time.time()
            print("[GRIPPER] 关闭指令下发..")

    def toggle(self):
        """切换夹爪开/关状态"""
        if self._target_state > 0.5:
            self.close()
        else:
            self.open()

    def get_state(self) -> float:
        """获取夹爪实际状态 (0.0~1.0)
        
        如果您有真实的电机编码器/GPIO反馈，请修改此方法使其直接读取真实开度。
        在没有真实反馈的情况下，我们这里通过时间插值模拟其物理闭合的过程，
        防止在 VLA 数据采集中出现 "夹爪刚下指令还未闭合，但 State 已经标为 0" 的错误对应。
        """
        elapsed = time.time() - self._last_toggle_time
        if elapsed >= self._actuation_time:
            return self._target_state

        # 线性插值模拟过程
        progress = elapsed / self._actuation_time
        return self._start_state + (self._target_state - self._start_state) * progress


# ═══════════════════════════════════════════════════════════════════════════
# 手柄 → 动作映射（复用仿真脚本的 GamepadMapper）
# ═══════════════════════════════════════════════════════════════════════════

class GamepadMapper:
    """将手柄数据转换为 7-DoF 笛卡尔 EE 动作 + 按钮事件。

    动作: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    所有输出范围 [-1, 1]，由外部缩放为实际增量。
    """

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.gripper_open = True
        self._prev_buttons: list[int] = []

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _button_pressed(self, buttons: list[int], idx: int) -> bool:
        """上升沿检测（0→1 时触发）"""
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return curr == 1 and prev == 0

    def map(self, data: dict | None) -> tuple[np.ndarray, dict]:
        """
        返回:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
            events: 按钮事件字典
        """
        action = np.zeros(7, dtype=np.float32)
        events = {
            "toggle_record": False,
            "discard": False,
            "toggle_gripper": False,
            "reset": False,
            "quit": False,
        }

        if data is None:
            self._prev_buttons = []
            action[6] = 1.0 if self.gripper_open else -1.0
            return action, events

        axes = data.get("axes", [])
        buttons = data.get("buttons", [])

        def axis(idx: int) -> float:
            return self._apply_deadzone(axes[idx]) if idx < len(axes) else 0.0

        # 笛卡尔 EE 控制
        action[0] = -axis(1)    # dx: 左摇杆前推 → +X
        action[1] = axis(0)     # dy: 左摇杆右推 → +Y
        action[2] = -axis(3)    # dz: 右摇杆上推 → +Z
        action[5] = axis(2)     # dyaw: 右摇杆右推 → yaw

        # 扳机 → pitch
        if len(axes) > 5:
            lt = (axes[4] + 1.0) / 2.0
            rt = (axes[5] + 1.0) / 2.0
            action[4] = rt - lt

        # 肩键 → roll
        lb_held = buttons[4] if len(buttons) > 4 else 0
        rb_held = buttons[5] if len(buttons) > 5 else 0
        action[3] = float(rb_held - lb_held)

        # 夹爪切换 (X)
        if self._button_pressed(buttons, 2):
            self.gripper_open = not self.gripper_open
            events["toggle_gripper"] = True

        action[6] = 1.0 if self.gripper_open else -1.0

        # 按钮事件
        events["toggle_record"] = self._button_pressed(buttons, 0)  # A
        events["discard"] = self._button_pressed(buttons, 1)        # B
        events["reset"] = self._button_pressed(buttons, 3)          # Y
        events["quit"] = self._button_pressed(buttons, 7)           # Start

        self._prev_buttons = list(buttons)
        return action, events


# ═══════════════════════════════════════════════════════════════════════════
# 相机管理器
# ═══════════════════════════════════════════════════════════════════════════

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
except ImportError:
    print("[WARNING] 未找到 pyorbbecsdk, 请安装。例如: pip install pyorbbecsdk")
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode = None, None, None, None, None


class CameraManager:
    """管理 Orbbec Femto Bolt 相机（通过 pyorbbecsdk），支持 D2C 对齐"""

    def __init__(self, cam_id: int = 0, width: int = 640, height: int = 480):
        self.cam_id = cam_id  # 对于单设备可能不使用
        self.width = width
        self.height = height
        self.pipeline = None

    def open(self):
        """打开相机并开启 D2C"""
        if Pipeline is None:
            raise RuntimeError("无法导入 pyorbbecsdk，无法使用 Orbbec 相机深度功能。")
            
        self.pipeline = Pipeline()
        config = Config()
        
        # 配置 Color 流
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(self.width, 0, OBFormat.RGB, 30)
            config.enable_stream(color_profile)
        except Exception as e:
            print(f"[相机] 颜色流配置失败: {e}")
            raise
            
        # 配置 Depth 流
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_video_stream_profile(self.width, 0, OBFormat.Y16, 30)
            config.enable_stream(depth_profile)
        except Exception as e:
            print(f"[相机] 深度流配置失败: {e}")
            raise

        # 开启 D2C 对齐 (硬件对齐)
        config.set_align_mode(OBAlignMode.HW_MODE)
        
        self.pipeline.start(config)
        print(f"[相机] Orbbec Femto Bolt 已启动并开启 D2C 深度对齐 ({self.width}x{self.height})")

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """读取一帧对齐的 RGB-D
        返回:
            rgb: (H, W, 3) uint8, RGB 格式, 或 None
            depth: (H, W) float32, 单位 mm, 或 None
        """
        if self.pipeline is None:
            return None, None

        # 等待一帧数据
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        # 提取 RGB 图像
        color_data = np.asanyarray(color_frame.get_data())
        rgb = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

        # 提取深度图像
        depth_data = np.asanyarray(depth_frame.get_data())
        # pyorbbecsdk 返回的 depth_data 是 1D array
        if depth_data.dtype == np.uint8:
            depth_data = depth_data.view(np.uint16)
        depth = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        return rgb, depth.astype(np.float32)

    def close(self):
        """释放相机"""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            print("[相机] 已释放")


# ═══════════════════════════════════════════════════════════════════════════
# CLI 参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DOBOT CR5 真机 VLA 数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", type=str, default="drag", choices=["drag", "gamepad"],
                    help="采集模式: drag=拖拽, gamepad=手柄 (默认: drag)")
    p.add_argument("--ip", type=str, default="192.168.50.102",
                    help="机器人 IP 地址 (默认: 192.168.50.102)")
    p.add_argument("--cam_id", type=int, default=0,
                    help="相机设备索引 (默认: 0)")
    p.add_argument("--cam_width", type=int, default=640, help="相机宽度 (默认: 640)")
    p.add_argument("--cam_height", type=int, default=480, help="相机高度 (默认: 480)")
    p.add_argument("--port", type=int, default=9876, help="手柄 UDP 端口 (默认: 9876)")
    p.add_argument("--hz", type=int, default=10, help="采集/控制频率 Hz (默认: 10)")
    p.add_argument("--deadzone", type=float, default=0.15, help="摇杆死区 (默认: 0.15)")
    p.add_argument("--image_size", type=int, default=224, help="保存图像尺寸 (默认: 224)")
    p.add_argument("--save_dir", type=str, default=None,
                    help="数据保存目录 (默认: data/vla_dataset_real/)")
    p.add_argument("--task", type=str, default="pour water from bottle into cup",
                    help="任务语言描述 (默认: pour water from bottle into cup)")
    p.add_argument("--min_steps", type=int, default=10,
                    help="有效 episode 最少步数 (默认: 10)")
    p.add_argument("--no_robot", action="store_true", default=False,
                    help="仅相机模式（不连接机器人）")
    p.add_argument("--speed_ratio", type=int, default=30,
                    help="机器人速度比例 1-100 (默认: 30)")

    # -- 手柄模式专用参数 --
    p.add_argument("--pos_scale", type=float, default=2.0,
                    help="[手柄模式] 位置增量缩放 mm/step (默认: 2.0)")
    p.add_argument("--rot_scale", type=float, default=1.0,
                    help="[手柄模式] 姿态增量缩放 °/step (默认: 1.0)")

    # -- 初始关节角度（度）--
    p.add_argument("--home_joints", type=float, nargs=6,
                    default=[0.0, 0.0, -90.0, 0.0, -90.0, 0.0],
                    help="初始关节角度（度）(默认: 0 0 -90 0 -90 0)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 状态向量构建
# ═══════════════════════════════════════════════════════════════════════════

def build_state_vector(robot: DobotCR5, gripper: GripperController) -> np.ndarray:
    """构建 19 维状态向量

    布局: [关节角(6) + 关节速度(6) + 夹爪状态(1) + 末端位姿(6)] = 19 维

    关节角: 度
    关节速度: 度/秒
    夹爪状态: 1.0=打开, 0.0=关闭
    末端位姿: [x(m), y(m), z(m), rx(°), ry(°), rz(°)]
    """
    state = np.zeros(19, dtype=np.float32)

    # 关节角度 (6)
    state[0:6] = robot.joint_angles

    # 关节速度 (6)
    state[6:12] = robot.actual_joint_speeds

    # 夹爪状态 (1)
    state[12] = gripper.get_state()

    # 末端位姿 (6): cartesian_pose 前3维是位置(m)，后3维是姿态(°)
    state[13:19] = robot.cartesian_pose

    return state


def build_dummy_state(gripper: GripperController) -> np.ndarray:
    """无机器人连接时的虚拟状态向量"""
    state = np.zeros(19, dtype=np.float32)
    state[12] = gripper.get_state()
    return state


# ═══════════════════════════════════════════════════════════════════════════
# 动作向量构建（用于 HDF5 记录）
# ═══════════════════════════════════════════════════════════════════════════

def normalize_angle_degrees(angle):
    """将角度限制在 [-180, 180] 范围内"""
    return (angle + 180) % 360 - 180


def build_action_from_state_delta(
    prev_state: np.ndarray, curr_state: np.ndarray, gripper: GripperController
) -> np.ndarray:
    """从状态差分构建 7-DoF 动作向量（拖拽模式用）

    动作: [dx, dy, dz, drx, dry, drz, gripper] = 7 维
    其中位置增量单位为 m, 姿态增量单位为 rad
    """
    action = np.zeros(7, dtype=np.float32)

    # 位置差分 (m)
    action[0:3] = curr_state[13:16] - prev_state[13:16]
    
    # 姿态差分 (rad) 并处理跨界问题
    action[3:6] = normalize_angle_degrees(curr_state[16:19] - prev_state[16:19]) * (np.pi / 180.0)

    # 夹爪: 将 0.0~1.0 的平滑状态映射到 -1.0~1.0
    action[6] = gripper.get_state() * 2.0 - 1.0

    return action


# ═══════════════════════════════════════════════════════════════════════════
# 拖拽模式主循环
# ═══════════════════════════════════════════════════════════════════════════

def run_drag_mode(args, robot: DobotCR5 | None, camera: CameraManager,
                  gripper: GripperController, recorder: EpisodeRecorder,
                  receiver: GamepadReceiver | None, save_dir: str):
    """拖拽模式：操作员手动拖动机器人采集数据"""

    # 任务语言池
    TASK_PROMPTS = _make_task_prompts(args.task)

    # 进入拖拽模式
    if robot is not None:
        print("[INFO] 正在进入拖拽模式...")
        robot.start_drag()
        time.sleep(0.5)
        print("[INFO] 已进入拖拽模式，可以手动拖动机器人")

    recording = False
    episode_count = 0
    control_dt = 1.0 / args.hz
    prev_state = None

    # OpenCV 窗口
    WIN_NAME = "真机采集 — 拖拽模式"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 500)

    print("\n操作说明：")
    print("  A / 键盘'r' = 切换录制 | B / 键盘'd' = 丢弃")
    print("  X / 键盘'g' = 切换夹爪 | Start / ESC = 退出")
    print("  等待操作...\n")

    try:
        while True:
            loop_t0 = time.time()

            # ---- 读取按钮事件 ----
            events = {
                "toggle_record": False,
                "discard": False,
                "toggle_gripper": False,
                "quit": False,
            }

            # 手柄事件（如果连接了手柄）
            gp_action = None
            if receiver is not None:
                gp_data = receiver.get_latest()
                if gp_data is not None:
                    # 只用按钮事件，不用摇杆动作
                    _mapper = getattr(run_drag_mode, '_mapper', None)
                    if _mapper is None:
                        _mapper = GamepadMapper(deadzone=args.deadzone)
                        run_drag_mode._mapper = _mapper
                    _, gp_events = _mapper.map(gp_data)
                    events["toggle_record"] = gp_events["toggle_record"]
                    events["discard"] = gp_events["discard"]
                    events["toggle_gripper"] = gp_events["toggle_gripper"]
                    events["quit"] = gp_events["quit"]

            # ---- 读取相机 ----
            rgb, depth = camera.read()

            # ---- 读取机器人状态 ----
            if robot is not None:
                curr_state = build_state_vector(robot, gripper)
            else:
                curr_state = build_dummy_state(gripper)

            # ---- 处理事件 ----
            if events["quit"]:
                print("\n[INFO] 退出请求")
                break

            if events["toggle_gripper"]:
                gripper.toggle()

            if events["toggle_record"]:
                if not recording:
                    recording = True
                    recorder.reset()
                    chosen_task = random.choice(TASK_PROMPTS)
                    recorder.task_description = chosen_task
                    prev_state = curr_state.copy()
                    print(f"[●REC] 开始录制 — episode {episode_count}")
                    print(f"       任务: \"{chosen_task}\"")
                else:
                    _save_episode(recorder, save_dir, episode_count, args)
                    episode_count += 1 if recorder.num_steps >= args.min_steps else 0
                    recording = False
                    recorder.reset()
                    prev_state = None

            if events["discard"]:
                if recording:
                    print(f"[✗丢弃] Episode 已丢弃 ({recorder.num_steps} 步)")
                    recorder.reset()
                    recording = False
                    prev_state = None

            # ---- 录制数据 ----
            if recording and rgb is not None:
                # 构建动作（状态差分）
                action = build_action_from_state_delta(prev_state, curr_state, gripper)
                prev_state = curr_state.copy()

                # 深度处理
                depth_for_record = depth
                if depth_for_record is None:
                    # 无深度时用全零占位
                    depth_for_record = np.zeros(
                        (rgb.shape[0], rgb.shape[1]), dtype=np.float32
                    )

                recorder.add_step(
                    rgb=rgb,
                    depth=depth_for_record,
                    state=curr_state,
                    action=action,
                    timestamp=time.time(),
                )

            # ---- 可视化 ----
            display = _build_display(
                rgb, depth, curr_state, recording, episode_count,
                recorder.num_steps, gripper, mode="拖拽"
            )
            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            # 键盘事件
            if key == 27:  # ESC
                print("\n[INFO] ESC 退出")
                break
            elif key == ord('r'):
                events["toggle_record"] = True
                # 重新触发录制逻辑（下一帧处理）
                if not recording:
                    recording = True
                    recorder.reset()
                    chosen_task = random.choice(TASK_PROMPTS)
                    recorder.task_description = chosen_task
                    prev_state = curr_state.copy()
                    print(f"[●REC] 开始录制 — episode {episode_count}")
                    print(f"       任务: \"{chosen_task}\"")
                else:
                    _save_episode(recorder, save_dir, episode_count, args)
                    episode_count += 1 if recorder.num_steps >= args.min_steps else 0
                    recording = False
                    recorder.reset()
                    prev_state = None
            elif key == ord('d'):
                if recording:
                    print(f"[✗丢弃] Episode 已丢弃 ({recorder.num_steps} 步)")
                    recorder.reset()
                    recording = False
                    prev_state = None
            elif key == ord('g'):
                gripper.toggle()

            # ---- 频率控制 ----
            elapsed = time.time() - loop_t0
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 中断")
        if recording and recorder.num_steps >= args.min_steps:
            _save_episode(recorder, save_dir, episode_count, args)

    # 退出拖拽模式
    if robot is not None:
        print("[INFO] 退出拖拽模式...")
        robot.stop_drag()
        time.sleep(0.3)

    cv2.destroyAllWindows()
    return episode_count


# ═══════════════════════════════════════════════════════════════════════════
# 手柄模式主循环
# ═══════════════════════════════════════════════════════════════════════════

def run_gamepad_mode(args, robot: DobotCR5 | None, camera: CameraManager,
                     gripper: GripperController, recorder: EpisodeRecorder,
                     receiver: GamepadReceiver, save_dir: str):
    """手柄模式：通过手柄遥操作，使用 MovL 笛卡尔运动"""

    TASK_PROMPTS = _make_task_prompts(args.task)
    mapper = GamepadMapper(deadzone=args.deadzone)

    recording = False
    episode_count = 0
    control_dt = 1.0 / args.hz

    WIN_NAME = "真机采集 — 手柄模式"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 800, 500)

    print("\n操作说明：")
    print("  左摇杆 = XY运动 | 右摇杆 = Z高度+Yaw旋转")
    print("  LT/RT = Pitch | LB/RB = Roll")
    print("  A = 切换录制 | B = 丢弃 | X = 夹爪 | Y = 复位 | Start = 退出")
    print("  等待手柄连接...\n")

    try:
        while True:
            loop_t0 = time.time()

            # ---- 读取手柄 ----
            gp_data = receiver.get_latest()
            action_np, events = mapper.map(gp_data)

            # ---- 处理事件 ----
            if events["quit"]:
                print("\n[INFO] 退出请求")
                break

            if events["reset"]:
                if recording:
                    print("[WARN] 录制停止并丢弃（复位）")
                    recorder.reset()
                    recording = False
                if robot is not None:
                    print("[INFO] 回到初始位姿...")
                    robot.joint_mov_j(list(args.home_joints))
                    time.sleep(3)
                    print("[INFO] 复位完成")
                continue

            if events["toggle_gripper"]:
                gripper.toggle()

            if events["toggle_record"]:
                if not recording:
                    recording = True
                    recorder.reset()
                    chosen_task = random.choice(TASK_PROMPTS)
                    recorder.task_description = chosen_task
                    print(f"[●REC] 开始录制 — episode {episode_count}")
                    print(f"       任务: \"{chosen_task}\"")
                else:
                    _save_episode(recorder, save_dir, episode_count, args)
                    episode_count += 1 if recorder.num_steps >= args.min_steps else 0
                    recording = False
                    recorder.reset()

            if events["discard"]:
                if recording:
                    print(f"[✗丢弃] Episode 已丢弃 ({recorder.num_steps} 步)")
                    recorder.reset()
                    recording = False

            # ---- 执行笛卡尔运动 (ServoP) ----
            has_motion = np.any(np.abs(action_np[:6]) > 1e-6)
            if has_motion and robot is not None:
                # 当前末端位姿（位置 m → mm, 姿态 °）
                curr_pose = list(robot.cartesian_pose)  # [x_m, y_m, z_m, rx, ry, rz]

                # 计算目标位姿（位置增量 mm, 姿态增量 °）
                target_x = curr_pose[0] * 1000.0 + action_np[0] * args.pos_scale
                target_y = curr_pose[1] * 1000.0 + action_np[1] * args.pos_scale
                target_z = curr_pose[2] * 1000.0 + action_np[2] * args.pos_scale
                target_rx = curr_pose[3] + action_np[3] * args.rot_scale
                target_ry = curr_pose[4] + action_np[4] * args.rot_scale
                target_rz = curr_pose[5] + action_np[5] * args.rot_scale

                robot.servo_p(target_x, target_y, target_z,
                              target_rx, target_ry, target_rz)

            # ---- 读取相机 ----
            rgb, depth = camera.read()

            # ---- 读取机器人状态 ----
            if robot is not None:
                curr_state = build_state_vector(robot, gripper)
            else:
                curr_state = build_dummy_state(gripper)

            # ---- 录制数据 ----
            if recording and rgb is not None:
                depth_for_record = depth if depth is not None else np.zeros(
                    (rgb.shape[0], rgb.shape[1]), dtype=np.float32
                )

                # 构造物理 Action [m, m, m, rad, rad, rad, gripper] 用于记录
                record_action = np.zeros(7, dtype=np.float32)
                record_action[0:3] = action_np[0:3] * (args.pos_scale / 1000.0)
                record_action[3:6] = action_np[3:6] * args.rot_scale * (np.pi / 180.0)
                record_action[6] = gripper.get_state() * 2.0 - 1.0

                recorder.add_step(
                    rgb=rgb,
                    depth=depth_for_record,
                    state=curr_state,
                    action=record_action,
                    timestamp=time.time(),
                )

            # ---- 可视化 ----
            display = _build_display(
                rgb, depth, curr_state, recording, episode_count,
                recorder.num_steps, gripper, mode="手柄",
                gp_connected=receiver.is_connected,
            )
            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            # ---- 频率控制 ----
            elapsed = time.time() - loop_t0
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 中断")
        if recording and recorder.num_steps >= args.min_steps:
            _save_episode(recorder, save_dir, episode_count, args)

    cv2.destroyAllWindows()
    return episode_count


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _make_task_prompts(default_task: str) -> list[str]:
    """生成任务语言提示池"""
    prompts = [
        default_task,
        "pour water from bottle into cup",
        "pour the water into the cup",
        "pick up the bottle and pour water into the cup",
        "fill the cup with water from the bottle",
        "grasp the bottle and pour its contents into the cup",
        "transfer water from the bottle to the cup",
        "pour water from the bottle into the target cup",
        "grab the bottle, move it over the cup, and pour",
    ]
    seen = set()
    return [p for p in prompts if not (p in seen or seen.add(p))]


def _save_episode(recorder: EpisodeRecorder, save_dir: str,
                  episode_count: int, args) -> bool:
    """保存 episode 并打印状态"""
    if recorder.num_steps >= args.min_steps:
        ep_path = next_episode_path(save_dir)
        recorder.save(ep_path, dt=1.0 / args.hz)
        print(f"[✓保存] Episode 已保存: {ep_path}  ({recorder.num_steps} 步)")
        return True
    else:
        print(f"[✗跳过] Episode 太短 ({recorder.num_steps} < {args.min_steps})，已丢弃")
        return False


def _build_display(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    state: np.ndarray,
    recording: bool,
    episode_count: int,
    rec_steps: int,
    gripper: GripperController,
    mode: str = "拖拽",
    gp_connected: bool | None = None,
) -> np.ndarray:
    """构建 OpenCV 可视化画面"""

    panel_h = 400

    # RGB 画面
    if rgb is not None:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = rgb_bgr.shape[:2]
        panel_w = int(w * panel_h / h)
        rgb_panel = cv2.resize(rgb_bgr, (panel_w, panel_h))
    else:
        panel_w = int(panel_h * 4 / 3)
        rgb_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(rgb_panel, "No Camera", (10, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    panels = [rgb_panel]

    # 深度画面
    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            if d_max > d_min:
                depth_norm[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        h, w = depth_color.shape[:2]
        dp_w = int(w * panel_h / h)
        depth_panel = cv2.resize(depth_color, (dp_w, panel_h))
        cv2.putText(depth_panel, "Depth", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        panels.append(depth_panel)

    display = np.hstack(panels)

    # HUD 叠加信息
    rec_text = f"[REC {rec_steps}]" if recording else "[IDLE]"
    grip_text = "OPEN" if gripper.get_state() > 0.5 else "CLOSED"
    ee_pose = state[13:19]

    hud_line1 = f"Mode:{mode}  {rec_text}  Ep:{episode_count}  Grip:{grip_text}"
    if gp_connected is not None:
        hud_line1 += f"  GP:{'OK' if gp_connected else '--'}"

    hud_line2 = (
        f"EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}] m  "
        f"[{ee_pose[3]:.1f}, {ee_pose[4]:.1f}, {ee_pose[5]:.1f}] deg"
    )

    color = (0, 255, 0) if recording else (200, 200, 200)
    cv2.putText(display, hud_line1, (10, display.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(display, hud_line2, (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # 录制红点
    if recording:
        cv2.circle(display, (display.shape[1] - 25, 25), 12, (0, 0, 255), -1)

    return display


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # 数据保存目录
    save_dir = args.save_dir or os.path.join(_REPO_ROOT, "data", "vla_dataset_real")

    # ── 打印配置 ──
    print("\n" + "=" * 64)
    print("DOBOT CR5 — 真机 VLA 数据采集")
    print("=" * 64)
    print(f"  采集模式       : {args.mode}")
    print(f"  机器人 IP      : {args.ip}")
    print(f"  相机设备索引   : {args.cam_id}")
    print(f"  采集频率       : {args.hz} Hz")
    print(f"  图像尺寸       : {args.image_size}×{args.image_size}")
    print(f"  保存目录       : {save_dir}")
    print(f"  任务描述       : {args.task}")
    print(f"  最少步数       : {args.min_steps}")
    print(f"  连接机器人     : {'否' if args.no_robot else '是'}")
    if args.mode == "gamepad":
        print(f"  位置增量       : {args.pos_scale} mm/step")
        print(f"  姿态增量       : {args.rot_scale} °/step")
        print(f"  手柄 UDP 端口  : {args.port}")
    print(f"  初始关节角度   : {args.home_joints}")
    print("=" * 64)

    # ── 连接机器人 ──
    robot = None
    if not args.no_robot:
        robot = DobotCR5(ip_address=args.ip)
        try:
            robot.connect()
            time.sleep(1)
            print(f"[INFO] 机器人模式: {robot.robot_mode}")

            robot.enable()
            time.sleep(2)
            print(f"[INFO] 使能后模式: {robot.robot_mode}")

            robot.set_speed_ratio(args.speed_ratio)
            print(f"[INFO] 速度比例: {args.speed_ratio}%")

            # 手柄模式：先运动到初始位姿
            if args.mode == "gamepad":
                print("[INFO] 运动到初始位姿...")
                robot.joint_mov_j(list(args.home_joints))
                time.sleep(5)
                print(f"[INFO] 当前关节角: {[round(a, 2) for a in robot.joint_angles]}")
                print(f"[INFO] 当前末端位姿: {[round(p, 4) for p in robot.cartesian_pose]}")

        except Exception as e:
            print(f"[ERROR] 机器人连接失败: {e}")
            robot = None
    else:
        print("[INFO] --no_robot 模式，不连接机器人")

    # ── 打开相机 ──
    camera = CameraManager(cam_id=args.cam_id, width=args.cam_width, height=args.cam_height)
    try:
        camera.open()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] 将继续运行但无相机画面")

    # ── 夹爪 ──
    gripper = GripperController()

    # ── 录制器 ──
    recorder = EpisodeRecorder(image_size=args.image_size, task_description=args.task)

    # ── 手柄接收器 ──
    receiver = None
    if args.mode == "gamepad" or True:  # 拖拽模式也可选用手柄按钮
        receiver = GamepadReceiver(port=args.port)
        receiver.start()

    # ── 运行主循环 ──
    try:
        if args.mode == "drag":
            episode_count = run_drag_mode(
                args, robot, camera, gripper, recorder, receiver, save_dir
            )
        else:
            if receiver is None:
                print("[ERROR] 手柄模式需要手柄连接")
                return
            episode_count = run_gamepad_mode(
                args, robot, camera, gripper, recorder, receiver, save_dir
            )
    finally:
        # ── 清理 ──
        if receiver is not None:
            receiver.stop()
        camera.close()
        if robot is not None:
            robot.disable()
            time.sleep(0.5)
            robot.disconnect()

    print(f"\n[INFO] 完成。共保存 {episode_count} 个 episodes")


if __name__ == "__main__":
    main()
