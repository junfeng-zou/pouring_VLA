#!/usr/bin/env python3
"""
PouringEnv 仿真端 — 仅 Isaac Lab，通过 TCP 把相机 JPEG 发给策略进程并接收 7 维动作。

与 `scripts/inference/openvla_lora_policy_client.py` 配对使用（先起本脚本并等待连接，再起客户端）。

默认仅在策略控制步与客户端交换图像/动作，中间仿真步本地用 Δ6=0 与上一拍夹爪命令推进，避免 OpenVLA 推理阻塞时整段物理挂起。若需旧版「每步 SKIP+recv」，请加 `--legacy_tcp_each_step`。

示例（注意：`isaaclab.sh` 在脚本路径后需加 `--`，否则 `--port` 会被交给 Kit 而非 Python）：
    /path/to/IsaacLab/isaaclab.sh -p scripts/inference/pouring_sim_policy_server.py -- --port 9875

    # 另一终端
    python scripts/inference/openvla_lora_policy_client.py --host 127.0.0.1 --port 9875 ...
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaacsim import SimulationApp

from tools.vla.openvla_lora_runtime import ensure_rgb_uint8_hwc
from tools.vla.pouring_policy_wire import handshake_server, recv_action7, send_obs_jpeg, send_skip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PouringEnv + TCP 策略桥接（仅仿真）")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=9875)
    p.add_argument("--hz", type=float, default=10.0, help="策略控制频率（与客户端一致）")
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0, help="0 = 不限")
    p.add_argument("--jpeg_quality", type=int, default=85, help="JPEG 质量 1–100")
    p.add_argument(
        "--action_timeout_s",
        type=float,
        default=120.0,
        help="等待客户端动作的 socket 超时（秒）；超时后退出",
    )
    p.add_argument(
        "--legacy_tcp_each_step",
        action="store_true",
        help="每步仿真都发 OIMG/SKIP 并阻塞等待 ACT7（旧行为）。默认仅在策略步通信，避免推理时仿真整段挂起",
    )
    p.add_argument(
        "--pos_action_scale",
        type=float,
        default=None,
        help="覆盖环境笛卡尔位置缩放（米/步，动作∈[-1,1] 时最大平移步长）。默认 0.005；想更快可试 0.01～0.02",
    )
    p.add_argument(
        "--rot_action_scale",
        type=float,
        default=None,
        help="覆盖姿态增量缩放（弧度/步）。默认 0.01",
    )
    p.add_argument(
        "--show_ee_frame",
        action="store_true",
        help="在末端 link_6 显示坐标系：X 红、Y 绿、Z 蓝（USD 箭头）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    sim_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
            "enable_cameras": True,
        }
    )

    import carb

    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

    import cv2

    from envs.pouring_env import PouringEnv, PouringEnvCfg

    cfg = PouringEnvCfg()
    cfg.scene.num_envs = 1
    if args.show_ee_frame:
        cfg.debug_visualize_ee_frame = True
    if args.pos_action_scale is not None:
        cfg.pos_action_scale = float(args.pos_action_scale)
    if args.rot_action_scale is not None:
        cfg.rot_action_scale = float(args.rot_action_scale)
    env = PouringEnv(cfg)
    print(
        f"[服务端] 动作缩放 pos_action_scale={cfg.pos_action_scale} m/step  "
        f"rot_action_scale={cfg.rot_action_scale} rad/step"
    )

    sim_dt = cfg.sim.dt * cfg.decimation
    control_interval = max(1, int(round(1.0 / (args.hz * sim_dt))))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    print(f"[服务端] 监听 {args.host}:{args.port}，等待策略客户端连接…")
    if args.legacy_tcp_each_step:
        print("[服务端] TCP: 每仿真步交换（--legacy_tcp_each_step）")
    else:
        print("[服务端] TCP: 仅策略步交换（默认；与当前 openvla_lora_policy_client 配对）")
    conn, addr = sock.accept()
    conn.settimeout(args.action_timeout_s)
    print(f"[服务端] 已连接: {addr}")
    try:
        handshake_server(conn)
    except Exception as exc:
        print(f"[服务端] 握手失败: {exc}")
        conn.close()
        sock.close()
        sim_app.close()
        return 1

    win = "Pouring policy bridge — camera"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    obs, _ = env.reset()
    step = 0
    steps_since_reset = 0
    last_policy_action7 = np.zeros(7, dtype=np.float32)

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                print("[服务端] 达到 max_steps，退出。")
                break

            run_policy = steps_since_reset > args.warmup_steps and (step % control_interval == 0)

            rgb_np: np.ndarray | None = None
            cam_data = env._camera.data
            if "rgb" in cam_data.output and cam_data.output["rgb"].numel() > 0:
                rgb = cam_data.output["rgb"][0].cpu().numpy()
                if rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]
                rgb_np = rgb

            # 相对 IK：非策略步用 Δ6=0；夹爪在 env 里按绝对 [-1,1] 映射，沿用上一拍策略命令以免漂移。
            action_np = np.zeros(7, dtype=np.float32)
            action_np[6] = last_policy_action7[6]

            try:
                if args.legacy_tcp_each_step:
                    if rgb_np is not None and run_policy:
                        rgb_u8 = ensure_rgb_uint8_hwc(rgb_np)
                        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                        ok, buf = cv2.imencode(
                            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
                        )
                        if ok:
                            send_obs_jpeg(conn, buf.tobytes())
                        else:
                            print("[服务端] JPEG 编码失败，发 SKIP")
                            send_skip(conn)
                    else:
                        send_skip(conn)

                    raw = recv_action7(conn)
                    action_np = np.frombuffer(raw, dtype=np.float32).copy()
                    if action_np.shape != (7,):
                        raise ValueError(f"动作形状 {action_np.shape}")
                    action_np = np.clip(action_np, -1.0, 1.0)
                    last_policy_action7 = action_np.copy()
                else:
                    need_exchange = rgb_np is not None and run_policy
                    if need_exchange:
                        rgb_u8 = ensure_rgb_uint8_hwc(rgb_np)
                        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                        ok, buf = cv2.imencode(
                            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
                        )
                        if ok:
                            send_obs_jpeg(conn, buf.tobytes())
                        else:
                            print("[服务端] JPEG 编码失败，发 SKIP")
                            send_skip(conn)
                        raw = recv_action7(conn)
                        action_np = np.frombuffer(raw, dtype=np.float32).copy()
                        if action_np.shape != (7,):
                            raise ValueError(f"动作形状 {action_np.shape}")
                        action_np = np.clip(action_np, -1.0, 1.0)
                        last_policy_action7 = action_np.copy()
                    # 非策略步不读写 socket，仿真按固定步长推进；客户端在 recv 上阻塞直至下一策略帧
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                print(f"[服务端] 与客户端交换失败，退出: {exc}")
                break

            action_t = torch.tensor(action_np, device=env.device, dtype=torch.float32).unsqueeze(0)
            obs, _r, terminated, truncated, _info = env.step(action_t)
            step += 1
            steps_since_reset += 1
            if bool(terminated[0]) or bool(truncated[0]):
                obs, _ = env.reset()
                steps_since_reset = 0

            if not args.headless and rgb_np is not None:
                cv2.imshow(
                    win,
                    cv2.cvtColor(ensure_rgb_uint8_hwc(rgb_np), cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[服务端] 按 q 退出。")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("[服务端] KeyboardInterrupt")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        try:
            conn.close()
        except Exception:
            pass
        sock.close()
        sim_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
