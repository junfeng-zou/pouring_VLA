"""
Dobot CR5 机器人控制类 - Python版本
功能: 通过 TCP/IP 控制 Dobot CR5 机器人
"""

import socket
import struct
import threading
import time
from typing import List, Tuple, Optional


class DobotCR5:
    """Dobot CR5 机器人控制类"""
    
    def __init__(self, ip_address='192.168.50.102', 
                 dashboard_port=29999, 
                 move_port=30003, 
                 feedback_port=30004):
        """
        初始化机器人连接参数
        
        Args:
            ip_address: 机器人 IP 地址
            dashboard_port: Dashboard 端口
            move_port: 运动控制端口
            feedback_port: 实时反馈端口
        """
        self.ip_address = ip_address
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.feedback_port = feedback_port
        
        # 连接状态
        self.is_connected = False
        
        # Socket 连接
        self.dashboard_client: Optional[socket.socket] = None
        self.move_client: Optional[socket.socket] = None
        self.feedback_client: Optional[socket.socket] = None
        
        # 机器人状态
        self.robot_mode = "Disconnected"
        self.current_speed_ratio = 0
        self.joint_angles = [0.0] * 6
        self.cartesian_pose = [0.0] * 6
        self.actual_joint_speeds = [0.0] * 6
        
        # 反馈数据缓冲
        self.state_data = bytearray(1440)
        self.feedback_thread: Optional[threading.Thread] = None
        self.feedback_running = False
        
    def connect(self):
        """连接到机器人"""
        if self.is_connected:
            print("Already connected.")
            return
        
        try:
            print(f"Connecting to robot at {self.ip_address}...")
            
            # 创建 TCP 连接
            self.dashboard_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dashboard_client.settimeout(5.0)
            self.dashboard_client.connect((self.ip_address, self.dashboard_port))
            
            self.move_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.move_client.settimeout(5.0)
            self.move_client.connect((self.ip_address, self.move_port))
            
            self.feedback_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.feedback_client.settimeout(5.0)
            self.feedback_client.connect((self.ip_address, self.feedback_port))
            
            # 启动反馈线程
            self.is_connected = True
            self.feedback_running = True
            self.feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
            self.feedback_thread.start()
            
            
            print("Successfully connected to the robot.")
            
        except Exception as e:
            self.is_connected = False
            raise ConnectionError(f"Failed to connect to the robot: {e}")
    
    def disconnect(self):
        """断开机器人连接"""
        if not self.is_connected:
            print("Already disconnected.")
            return
        
        try:
            # 停止反馈线程
            self.feedback_running = False
            if self.feedback_thread:
                self.feedback_thread.join(timeout=1.0)
            
            # 关闭连接
            if self.dashboard_client:
                self.dashboard_client.close()
            if self.move_client:
                self.move_client.close()
            if self.feedback_client:
                self.feedback_client.close()
            
            self.is_connected = False
            self.robot_mode = "Disconnected"
            print("Disconnected from the robot.")
            
        except Exception as e:
            print(f"Error during disconnect: {e}")
    
    def enable(self, load=5.0):
        """使能机器人"""
        cmd = f"EnableRobot({load})"
        return self._send_dashboard_command(cmd)
    
    def disable(self):
        """下使能机器人"""
        return self._send_dashboard_command("DisableRobot()")
    
    def clear_error(self):
        """清除错误报警"""
        return self._send_dashboard_command("ClearError()")
    
    def reset(self):
        """复位机器人"""
        return self._send_dashboard_command("ResetRobot()")
    
    def set_speed_ratio(self, ratio: int):
        """设置速度比例 (1-100)"""
        cmd = f"SpeedFactor({ratio})"
        return self._send_dashboard_command(cmd)
    
    def get_robot_mode(self) -> str:
        """通过 Dashboard 查询机器人模式"""
        response = self._send_dashboard_command("RobotMode()")
        return response
    
    def get_pose(self) -> str:
        """通过 Dashboard 查询当前位姿"""
        response = self._send_dashboard_command("GetPose()")
        return response
    
    def get_angle(self) -> str:
        """通过 Dashboard 查询当前关节角度"""
        response = self._send_dashboard_command("GetAngle()")
        return response

    def get_joint_angles(self) -> List[float]:
        """实时反馈中的关节角（度），线程安全快照。"""
        return list(self.joint_angles)

    def get_actual_joint_speeds(self) -> List[float]:
        """实时反馈中的关节速度。"""
        return list(self.actual_joint_speeds)

    def get_cartesian_pose(self) -> List[float]:
        """实时反馈笛卡尔位姿 [x,y,z, rx,ry,rz]：位置 m，姿态度（与 ServoP 输入一致）。"""
        return list(self.cartesian_pose)
    
    def joint_mov_j(self, joints: List[float]):
        """关节运动（角度单位：度）"""
        cmd = f"JointMovJ({joints[0]},{joints[1]},{joints[2]},{joints[3]},{joints[4]},{joints[5]})"
        self._send_move_command(cmd)
    
    def servo_j(self, joints: List[float], t: float, lookahead_time: float, gain: float):
        """
        ServoJ 运动控制
        
        Args:
            joints: 目标关节角度（度）
            t: 指令周期（秒）
            lookahead_time: 前瞻时间（秒）
            gain: 增益
        """
        cmd = f"ServoJ({joints[0]},{joints[1]},{joints[2]},{joints[3]},{joints[4]},{joints[5]},t={t},lookahead_time={lookahead_time},gain={gain})"
        self._send_move_command(cmd)
    
    def mov_l(self, x: float, y: float, z: float, rx: float, ry: float, rz: float):
        """笛卡尔空间直线运动
        
        Args:
            x, y, z: 目标位置（mm）
            rx, ry, rz: 目标姿态（度）
        """
        cmd = f"MovL({x},{y},{z},{rx},{ry},{rz})"
        self._send_move_command(cmd)

    def servo_p(self, x: float, y: float, z: float, rx: float, ry: float, rz: float):
        """笛卡尔空间伺服运动
        
        Args:
            x, y, z: 目标位置（mm）
            rx, ry, rz: 目标姿态（度）
        """
        cmd = f"ServoP({x},{y},{z},{rx},{ry},{rz})"
        self._send_move_command(cmd)

    def start_drag(self):
        """进入拖拽（协作）模式"""
        return self._send_dashboard_command("StartDrag()")

    def stop_drag(self):
        """退出拖拽模式"""
        return self._send_dashboard_command("StopDrag()")

    def stop_move(self):
        """停止运动"""
        return self._send_move_command("StopScript()")
    
    def _send_dashboard_command(self, command: str) -> str:
        """发送 Dashboard 指令"""
        if not self.is_connected or not self.dashboard_client:
            raise ConnectionError("Not connected. Cannot send dashboard command.")
        
        try:
            self.dashboard_client.sendall(command.encode('utf-8'))
            time.sleep(0.002)
            
            response = ""
            try:
                self.dashboard_client.settimeout(0.1)
                response = self.dashboard_client.recv(1024).decode('utf-8')
            except socket.timeout:
                pass
            
            print(f"Sent(Dash): {command} | Rcvd: {response.strip()}")
            return response
            
        except Exception as e:
            raise RuntimeError(f"Failed to send dashboard command '{command}': {e}")
    
    def _send_move_command(self, command: str) -> str:
        """发送运动控制指令"""
        if not self.is_connected or not self.move_client:
            raise ConnectionError("Not connected. Cannot send move command.")
        
        try:
            self.move_client.sendall(command.encode('utf-8'))
            
            response = ""
            try:
                self.move_client.settimeout(0.01)
                response = self.move_client.recv(1024).decode('utf-8')
            except socket.timeout:
                pass
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Failed to send move command '{command}': {e}")
    
    def _feedback_loop(self):
        """实时反馈数据接收线程"""
        buffer = bytearray()  # 数据缓冲区
        while self.feedback_running and self.is_connected:
            try:
                if not self.feedback_client:
                    break
                
                self.feedback_client.settimeout(1.0)
                data = self.feedback_client.recv(4096)  # 增大接收缓冲区
                
                if data:
                    buffer.extend(data)
                    
                    # 当缓冲区有足够数据时解析
                    while len(buffer) >= 1440:
                        self.state_data = bytearray(buffer[:1440])
                        buffer = buffer[1440:]  # 移除已处理的数据
                        self._parse_feedback_data()
                else:
                    print("[DEBUG] Received empty data")
                    
            except socket.timeout:
                print("[DEBUG] Feedback socket timeout")
                continue
            except Exception as e:
                if self.feedback_running:
                    print(f"Feedback loop error: {e}")
                break
    
    def _parse_feedback_data(self):
        """解析反馈数据"""
        try:
            # 机器人模式 (字节位置 24, 从0开始索引)
            robot_mode_val = self.state_data[24]
            mode_map = {
                1: "INIT", 2: "BRAKE_OPEN", 3: "POWER_OFF",
                4: "DISABLED", 5: "ENABLE", 6: "BACKDRIVE",
                7: "RUNNING", 8: "RECORDING", 9: "ERROR",
                10: "PAUSE", 11: "JOG"
            }
            self.robot_mode = mode_map.get(robot_mode_val, "UNKNOWN")
            
            # 当前速度比例 (字节位置 64)
            self.current_speed_ratio = self._bytes_to_double(64)
            
            # 关节角度 (字节位置 432 开始)
            for i in range(6):
                self.joint_angles[i] = self._bytes_to_double(432 + i * 8)
            
            # 笛卡尔位姿 (字节位置 624 开始)
            for i in range(6):
                value = self._bytes_to_double(624 + i * 8)
                # 前3个是位置 (X, Y, Z)，从 mm 转换为 m
                if i < 3:
                    self.cartesian_pose[i] = value / 1000.0
                else:
                    self.cartesian_pose[i] = value
            
            # 关节速度 (字节位置 480 开始)
            for i in range(6):
                self.actual_joint_speeds[i] = self._bytes_to_double(480 + i * 8)
                
        except Exception as e:
            pass  # 静默处理解析错误
    
    def _bytes_to_double(self, start_index: int) -> float:
        """将 8 个字节转换为 double"""
        try:
            bytes_data = self.state_data[start_index:start_index + 8]
            return struct.unpack('d', bytes_data)[0]
        except:
            return 0.0
    
    def __del__(self):
        """析构函数"""
        if self.is_connected:
            self.disconnect()


if __name__ == "__main__":
    # 测试代码
    robot = DobotCR5(ip_address='192.168.50.104')  # 使用您的机器人 IP
    try:
        robot.connect()
        time.sleep(3)
        
        print("\n--- Feedback Port Data ---")
        print(f"Robot mode (from feedback): {robot.robot_mode}")
        print(f"Joint angles (from feedback): {[round(a, 4) for a in robot.joint_angles]}")
        print(f"Cartesian pose (from feedback): {[round(p, 4) for p in robot.cartesian_pose]}")
        print("=" * 50)
        
        robot.disconnect()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()