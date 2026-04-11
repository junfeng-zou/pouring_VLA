#!/usr/bin/env python3
"""
将 URDF/XML 发布到 /robot_description（std_msgs/String），供 RViz2 RobotModel（Description Source: Topic）使用。

robot_state_publisher 仍由 launch 单独传入 robot_description 参数；本节点仅负责话题，
与 RViz 默认订阅方式对齐。

用法:
  python3 scripts/robot/rviz_replay/robot_description_topic_pub.py --file /path/to/robot.urdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, required=True, help="URDF/XML 文件路径")
    p.add_argument("--hz", type=float, default=5.0, help="重复发布频率（Volatile QoS 下便于晚启动的 RViz 收到）")
    return p.parse_args()


class Pub(Node):
    def __init__(self, xml: str, hz: float):
        super().__init__("robot_description_topic_pub")
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(String, "/robot_description", qos)
        self._msg = String(data=xml)
        self._pub.publish(self._msg)
        period = 1.0 / max(hz, 0.2)
        self.create_timer(period, self._tick)
        self.get_logger().info("发布 /robot_description (String)，长度 %d 字符" % len(xml))

    def _tick(self) -> None:
        self._pub.publish(self._msg)


def main() -> int:
    args = parse_args()
    path = Path(args.file).expanduser().resolve()
    xml = path.read_text(encoding="utf-8")
    rclpy.init()
    node = Pub(xml, hz=args.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
