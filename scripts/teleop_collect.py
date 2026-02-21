#!/usr/bin/env python3
"""
DOBOT Nova 5 — Gamepad Teleoperation & VLA Data Collection
============================================================

Launch with:
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/teleop_collect.py

    # headless:
    /home/zjf/IsaacLab/isaaclab.sh -p scripts/teleop_collect.py --headless

Then on the local PC (with gamepad):
    python tools/gamepad_sender.py --ip <3090_TAILSCALE_IP>

Gamepad mapping (Xbox / generic dual-stick) — Cartesian EE control:
    Left  stick X           → EE Y  (left/right)
    Left  stick Y           → EE X  (forward/backward)
    Right stick X           → EE yaw rotation
    Right stick Y           → EE Z  (up/down)
    LT            (axis 4)  → pitch–
    RT            (axis 5)  → pitch+
    LB  (btn 4)             → roll–
    RB  (btn 5)             → roll+
    A   (btn 0)             → toggle recording
    B   (btn 1)             → discard current episode
    X   (btn 2)             → toggle gripper open/close
    Y   (btn 3)             → reset environment
    Start (btn 7)           → quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# -- Make project root importable --
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacsim import SimulationApp


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gamepad teleoperation for VLA data collection")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--port", type=int, default=9876, help="UDP port for gamepad receiver")
    p.add_argument("--hz", type=int, default=10, help="Control / recording frequency (Hz)")
    p.add_argument("--deadzone", type=float, default=0.15, help="Joystick deadzone")
    p.add_argument("--action_scale", type=float, default=0.01, help="Joint delta scale (rad/step)")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Dataset save directory (default: data/vla_dataset/)")
    p.add_argument("--image_size", type=int, default=224, help="Saved image size (square)")
    p.add_argument("--debug", action="store_true", default=False, help="Enable debug prints")
    return p.parse_args()


# ── Gamepad → Action mapping ────────────────────────────────────────────────

class GamepadMapper:
    """Convert raw gamepad data to a 7-DoF Cartesian EE action + button events.

    Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    All outputs are in [-1, 1] (env applies the actual scaling).
    """

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.gripper_open = True  # start open

        # Button edge detection (previous state)
        self._prev_buttons: list[int] = []

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _button_pressed(self, buttons: list[int], idx: int) -> bool:
        """True on the rising edge (was 0, now 1)."""
        if idx >= len(buttons):
            return False
        curr = buttons[idx]
        prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
        return curr == 1 and prev == 0

    def map(self, data: dict | None) -> tuple[np.ndarray, dict]:
        """
        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    All in [-1, 1]. gripper: +1=open, -1=closed
            events: dict of boolean flags for button presses
        """
        action = np.zeros(7, dtype=np.float32)
        events = {
            "toggle_record": False,
            "discard": False,
            "toggle_gripper": False,
            "reset_env": False,
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

        # ---- Cartesian EE control ----
        # Left stick → XY translation (robot base frame)
        #   stick X → EE Y (left/right)
        #   stick Y → EE X (forward/backward, inverted)
        action[0] = -axis(1)    # dx: push stick forward → EE moves +X
        action[1] = axis(0)     # dy: push stick right  → EE moves +Y

        # Right stick → Z height + yaw rotation
        #   stick Y → EE Z (up/down, inverted)
        #   stick X → EE yaw (rotate around Z axis)
        action[2] = -axis(3)    # dz:   push stick up    → EE moves +Z
        action[5] = axis(2)     # dyaw: push stick right → yaw CW

        # Triggers → pitch rotation
        #   LT/RT rest at -1.0, pressed → +1.0
        if len(axes) > 5:
            lt = (axes[4] + 1.0) / 2.0  # 0..1
            rt = (axes[5] + 1.0) / 2.0  # 0..1
            action[4] = rt - lt          # dpitch

        # Shoulder buttons → roll rotation
        lb_held = buttons[4] if len(buttons) > 4 else 0
        rb_held = buttons[5] if len(buttons) > 5 else 0
        action[3] = float(rb_held - lb_held)  # droll

        # ---- Gripper toggle (X button) ----
        if self._button_pressed(buttons, 2):
            self.gripper_open = not self.gripper_open
            events["toggle_gripper"] = True

        action[6] = 1.0 if self.gripper_open else -1.0

        # ---- Button events (edge-triggered) ----
        events["toggle_record"] = self._button_pressed(buttons, 0)  # A
        events["discard"] = self._button_pressed(buttons, 1)        # B
        events["reset_env"] = self._button_pressed(buttons, 3)      # Y
        events["quit"] = self._button_pressed(buttons, 7)           # Start

        # Save for next edge detection
        self._prev_buttons = list(buttons)

        return action, events


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve save directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = args.save_dir or os.path.join(project_dir, "data", "vla_dataset")

    # ── Launch SimulationApp ──
    sim_app = SimulationApp({
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "enable_cameras": True,
    })

    import carb
    carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

    # ── Isaac Lab imports (after SimulationApp) ──
    from envs.pouring_env import PouringEnv, PouringEnvCfg
    from tools.gamepad_receiver import GamepadReceiver
    from tools.vla_data_writer import EpisodeRecorder, next_episode_path

    # ── Environment ──
    cfg = PouringEnvCfg()
    cfg.scene.num_envs = 1
    env = PouringEnv(cfg)

    # ── Gamepad receiver ──
    receiver = GamepadReceiver(port=args.port)
    receiver.start()

    # ── Mapper & recorder ──
    mapper = GamepadMapper(deadzone=args.deadzone)
    recorder = EpisodeRecorder(image_size=args.image_size)

    # ── State ──
    recording = False
    episode_count = 0
    sim_dt = cfg.sim.dt * cfg.decimation
    control_interval = max(1, int(1.0 / (args.hz * sim_dt)))  # sim steps per control step

    print("\n" + "=" * 64)
    print("DOBOT Nova 5 — Gamepad Teleoperation")
    print("=" * 64)
    print(f"  UDP port         : {args.port}")
    print(f"  Control freq     : {args.hz} Hz  (every {control_interval} sim steps)")
    print(f"  Deadzone         : {args.deadzone}")
    print(f"  Action scale     : {args.action_scale} rad/step")
    print(f"  Image size       : {args.image_size}×{args.image_size}")
    print(f"  Save directory   : {save_dir}")
    print("=" * 64)
    print("  A = toggle recording | B = discard | X = gripper | Y = reset | Start = quit")
    print("  Waiting for gamepad data...\n")

    # ── OpenCV visualization ──
    import cv2

    # ── Reset ──
    obs, info = env.reset()
    step = 0
    t = 0.0

    try:
        while True:
            # ---------- Read gamepad ----------
            gp_data = receiver.get_latest()

            # ===== DEBUG POINT 1: Raw gamepad data =====
            if args.debug and step % 30 == 0:
                if gp_data is None:
                    print("[DEBUG-1] gp_data = None (no UDP packets received yet)")
                else:
                    axes = gp_data.get('axes', [])
                    buttons = gp_data.get('buttons', [])
                    seq = gp_data.get('seq', '?')
                    print(f"[DEBUG-1] seq={seq}  axes({len(axes)})={[round(a,3) for a in axes]}  buttons({len(buttons)})={buttons}")

            action_np, events = mapper.map(gp_data)

            # ===== DEBUG POINT 2: Mapped action =====
            if args.debug and step % 30 == 0:
                nonzero = any(abs(a) > 1e-6 for a in action_np[:6])
                print(f"[DEBUG-2] action=[{', '.join(f'{a:+.5f}' for a in action_np)}]  nonzero_arm={nonzero}  gripper={'OPEN' if action_np[6]>0 else 'CLOSED'}")

            # ---------- Handle events ----------
            if events["quit"]:
                print("\n[INFO] Quit requested via gamepad.")
                break

            if events["reset_env"]:
                if recording:
                    print("[WARN] Recording stopped & discarded due to reset.")
                    recorder.reset()
                    recording = False
                obs, info = env.reset()
                mapper.gripper_open = True  # reset gripper state to open
                step = 0
                t = 0.0
                print("[INFO] Environment reset.")
                continue

            if events["toggle_record"]:
                if not recording:
                    recording = True
                    recorder.reset()
                    print(f"[●REC] Recording started — episode {episode_count}")
                else:
                    # Save episode
                    ep_path = next_episode_path(save_dir)
                    recorder.save(ep_path, dt=1.0 / args.hz)
                    print(f"[✓SAVE] Episode saved: {ep_path}  ({recorder.num_steps} steps)")
                    episode_count += 1
                    recording = False
                    recorder.reset()

            if events["discard"]:
                if recording:
                    print(f"[✗DISCARD] Episode discarded ({recorder.num_steps} steps)")
                    recorder.reset()
                    recording = False

            if events["toggle_gripper"]:
                state = "OPEN" if mapper.gripper_open else "CLOSED"
                print(f"[GRIPPER] {state}")

            # ---------- Step environment ----------
            action_tensor = torch.tensor(action_np, device=env.device, dtype=torch.float32).unsqueeze(0)

            # ===== DEBUG POINT 3: Joint targets before & after step =====
            if args.debug and step % 30 == 0:
                jpos_before = env._robot.data.joint_pos[0].cpu().numpy()
                jtarget_before = env.robot_dof_targets[0].cpu().numpy()
                print(f"[DEBUG-3] PRE-STEP  joint_pos =[{', '.join(f'{j:+.4f}' for j in jpos_before[:6])}]")
                print(f"[DEBUG-3] PRE-STEP  dof_target=[{', '.join(f'{j:+.4f}' for j in jtarget_before[:6])}]")

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            t += sim_dt

            # ===== DEBUG POINT 3b: Joint targets after step =====
            if args.debug and step % 30 == 0:
                jtarget_after = env.robot_dof_targets[0].cpu().numpy()
                jpos_after = env._robot.data.joint_pos[0].cpu().numpy()
                print(f"[DEBUG-3] POST-STEP dof_target=[{', '.join(f'{j:+.4f}' for j in jtarget_after[:6])}]")
                print(f"[DEBUG-3] POST-STEP joint_pos =[{', '.join(f'{j:+.4f}' for j in jpos_after[:6])}]")
                print(f"[DEBUG-3] target_delta        =[{', '.join(f'{(jtarget_after[i]-jtarget_before[i]):+.5f}' for i in range(6))}]")
                print()

            # ---------- Record data ----------
            if recording:
                cam_data = env._camera.data
                if "rgb" in cam_data.output and "distance_to_image_plane" in cam_data.output:
                    rgb = cam_data.output["rgb"][0].cpu().numpy()          # (H, W, 4) or (H, W, 3)
                    if rgb.shape[-1] == 4:
                        rgb = rgb[:, :, :3]  # drop alpha
                    depth = cam_data.output["distance_to_image_plane"][0].cpu().numpy()  # (H, W, 1)
                    depth = depth.squeeze(-1)  # (H, W)

                    state_vec = obs["policy"][0].cpu().numpy()  # (20,)
                    recorder.add_step(
                        rgb=rgb,
                        depth=depth,
                        state=state_vec,
                        action=action_np,
                        timestamp=t,
                    )

            # ---------- Visualization ----------
            if step % 6 == 0:
                cam_data = env._camera.data
                if "rgb" in cam_data.output:
                    rgb_vis = cam_data.output["rgb"][0].cpu().numpy()
                    if rgb_vis.shape[-1] == 4:
                        rgb_vis = rgb_vis[:, :, :3]
                    rgb_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

                    # Resize for display
                    h, w = rgb_bgr.shape[:2]
                    display = cv2.resize(rgb_bgr, (w // 4, h // 4))

                    # HUD overlay
                    rec_text = f"[REC {recorder.num_steps}]" if recording else "[IDLE]"
                    gp_text = "GP:OK" if receiver.is_connected else "GP:--"
                    ee_pos = obs["policy"][0, 14:17].cpu().numpy()
                    hud = f"{rec_text}  Ep:{episode_count}  {gp_text}  EE:[{ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f}]"
                    cv2.putText(display, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if recording else (200, 200, 200), 1, cv2.LINE_AA)

                    cv2.imshow("Teleop — DOBOT Nova 5", display)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break

            # ---------- Print status ----------
            if step % (args.hz * 5) == 0 and step > 0:
                ee_pos = obs["policy"][0, 14:17].cpu().numpy()
                rec_str = f"●REC({recorder.num_steps})" if recording else "○IDLE"
                print(
                    f"  [t={t:6.1f}s]  {rec_str}  "
                    f"ep={episode_count}  "
                    f"EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]  "
                    f"grip={'O' if mapper.gripper_open else 'X'}"
                )

            step += 1

            # Handle resets from env
            if terminated.any() or truncated.any():
                if recording:
                    ep_path = next_episode_path(save_dir)
                    recorder.save(ep_path, dt=1.0 / args.hz)
                    print(f"[✓SAVE] Auto-saved on episode end: {ep_path}  ({recorder.num_steps} steps)")
                    episode_count += 1
                    recording = False
                    recorder.reset()
                obs, info = env.reset()
                step = 0
                t = 0.0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        if recording and recorder.num_steps > 0:
            ep_path = next_episode_path(save_dir)
            recorder.save(ep_path, dt=1.0 / args.hz)
            print(f"[✓SAVE] Saved on interrupt: {ep_path}  ({recorder.num_steps} steps)")

    # ── Cleanup ──
    cv2.destroyAllWindows()
    receiver.stop()
    env.close()
    sim_app.close()
    print(f"\n[INFO] Done. Total episodes saved: {episode_count}")


if __name__ == "__main__":
    main()
