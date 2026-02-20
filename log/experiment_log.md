1 在机械臂末端直接添加末端执行器时，会出现机械臂乱动的情况。
原因：isaac sim自带的末端执行器也包括Articulation，会和机械臂的Articulation产生冲突，导致机械臂乱动。
解决办法：将末端执行器添加到机械臂的usd文件中，作为机械臂的一部分。

2 对于isaac sim中的相机，设置的四元数朝向和实际的UI中的朝向四元数不一致。
原因： Isaac Lab 的 CameraCfg 在内部会做坐标系约定转换。设置的 convention="world" 表示：
你的输入：世界坐标系（前方 = +X，上方 = +Z）
USD Camera 内部：OpenGL 坐标系（前方 = -Z，上方 = +Y）
解决办法：需根据坐标变换关系调整四元数。