"""
Dobot Nova 5 — RViz2 可视化启动：静态 TF(map→dummy_link) + /robot_description 话题 + robot_state_publisher。

默认 URDF：scripts/robot/rviz_replay/urdf/nova5_robot.urdf。可与 hdf5_rviz_replay.py 配合回放 HDF5 关节轨迹。

用法（仓库根目录）::
    source /opt/ros/humble/setup.bash
    ros2 launch scripts/robot/rviz_replay/launch/rviz_nova5_replay.launch.py

换 CR5 模型::
    ros2 launch scripts/robot/rviz_replay/launch/rviz_nova5_replay.launch.py \\
        urdf:=$(pwd)/scripts/robot/rviz_replay/urdf/cr5_robot.urdf

另开终端回放 HDF5::
    python3 scripts/robot/rviz_replay/hdf5_rviz_replay.py --episode <path/to/episode.hdf5> --mode joints
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_urdf(urdf_path: Path, mesh_dir: str) -> str:
    xml = urdf_path.read_text(encoding="utf-8")
    mesh_dir = mesh_dir.strip()
    if mesh_dir:
        base = Path(mesh_dir).expanduser().resolve()
        if not base.is_dir():
            raise FileNotFoundError(f"mesh_dir 不是目录: {base}")
        prefix = base.as_uri()
        if not prefix.endswith("/"):
            prefix += "/"
        xml = xml.replace("package://dobot_rviz/meshes/", prefix)
    return xml


def generate_launch_description() -> LaunchDescription:
    repo = _repo_root()
    rviz_pkg = repo / "scripts" / "robot" / "rviz_replay"
    default_urdf = str(rviz_pkg / "urdf" / "nova5_robot.urdf")
    default_rviz = str(rviz_pkg / "launch" / "dobot_arm_replay.rviz")

    urdf_arg = DeclareLaunchArgument(
        "urdf",
        default_value=default_urdf,
        description="URDF 路径（默认 nova5_robot.urdf；可改为 cr5_robot.urdf）",
    )
    mesh_arg = DeclareLaunchArgument(
        "mesh_dir",
        default_value="",
        description="可选：STL 根目录，将 package://dobot_rviz/meshes/ 替换为该路径（内含 nova5/ 等）",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="是否启动 rviz2（默认加载 dobot_arm_replay.rviz，Fixed Frame=map）",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="RViz2 配置文件路径",
    )
    root_frame_arg = DeclareLaunchArgument(
        "tf_root_link",
        default_value="dummy_link",
        description="静态 TF map 的子坐标系：与 URDF 根 link 一致（nova5/cr5 均为 dummy_link）",
    )

    def _setup(context, *args, **kwargs):
        urdf_lc = LaunchConfiguration("urdf").perform(context)
        mesh_lc = LaunchConfiguration("mesh_dir").perform(context)
        use_rviz_lc = LaunchConfiguration("use_rviz").perform(context).lower() in (
            "1",
            "true",
            "yes",
        )
        rviz_cfg_lc = LaunchConfiguration("rviz_config").perform(context)
        root_lc = LaunchConfiguration("tf_root_link").perform(context)

        p = Path(urdf_lc).expanduser()
        if not p.is_absolute():
            p = (repo / p).resolve()
        robot_desc = _load_urdf(p, mesh_lc)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".urdf",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(robot_desc)
            urdf_tmp_path = tmp.name

        pub_script = str(repo / "scripts" / "robot" / "rviz_replay" / "robot_description_topic_pub.py")
        rviz_cfg = Path(rviz_cfg_lc).expanduser()
        if not rviz_cfg.is_absolute():
            rviz_cfg = (repo / rviz_cfg).resolve()

        nodes = [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_to_robot_root",
                arguments=[
                    "--frame-id",
                    "map",
                    "--child-frame-id",
                    root_lc,
                ],
                output="screen",
            ),
            Node(
                executable="python3",
                arguments=[pub_script, "--file", urdf_tmp_path],
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_desc}],
                output="screen",
            ),
        ]
        if use_rviz_lc:
            nodes.append(
                Node(
                    package="rviz2",
                    executable="rviz2",
                    arguments=["-d", str(rviz_cfg)],
                    output="screen",
                )
            )
        return nodes

    return LaunchDescription(
        [
            urdf_arg,
            mesh_arg,
            use_rviz_arg,
            rviz_config_arg,
            root_frame_arg,
            OpaqueFunction(function=_setup),
        ]
    )
