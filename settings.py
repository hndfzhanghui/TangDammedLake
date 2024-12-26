import json
import numpy as np

class SimulationSettings:
    def __init__(self, config_file):
        # 加载配置文件
        with open(config_file, 'r') as file:
            config = json.load(file)

        # 基础模拟参数
        self.time_step = config.get("time_step", 0.01)
        self.gravity = config.get("gravity", 9.81)
        self.grid_width_x = config.get("grid_width_x", 1.0)  # 网格X方向的宽度（单位：米）
        self.grid_width_y = config.get("grid_width_y", 1.0)  # 网格Y方向的宽度（单位：米）

        # 侵蚀和沉积参数
        self.erosion_constant = config.get("erosion_constant", 0.01)
        self.deposition_constant = config.get("deposition_constant", 0.01)

        # 蒸发参数
        self.evaporation_constant = config.get("evaporation_constant", 0.001)

        # 降雨和河流参数
        self.rainfall_intensity_mm_per_hour = config.get("rainfall_intensity_mm_per_hour", 0.02)  # 降雨强度（单位：米/秒）
        self.rainfall_probability = config.get("rainfall_probability", 0.1)  # 降雨发生的概率
        self.river_speed = config.get("river_speed", 1.0)  # 河流流速（单位：米/秒）

        # 管道模型参数
        self.min_pipe_area = config.get("min_pipe_area", 0.01)  # 管道最小截面积
        self.max_pipe_area = config.get("max_pipe_area", 100.0)  # 管道最大截面积

        # 初始条件
        self.initial_fluid_height = config.get("initial_fluid_height", 0.1)  # 初始水深

        # 其他流体属性
        self.viscosity = config.get("viscosity", 0.01)
        self.max_flow_speed = config.get("max_flow_speed", 10.0)

        # 边界条件
        self.boundary_conditions = config.get("boundary_conditions", {
            "left": "closed",
            "right": "closed",
            "top": "closed",
            "bottom": "closed"
        })

        # 加载地形文件
        self.terrain_file = config.get("terrain_file", "terrain_data.txt")  # 地形数据文件
        self.terrain_height = np.loadtxt(self.terrain_file)  # 假设文件为TXT格式，存储地形高度数组

        # 由terrain_file的shape决定 
        self.grid_size_x = self.terrain_height.shape[1]  # 网格X方向大小
        self.grid_size_y = self.terrain_height.shape[0]  # 网格Y方向大小

        # 加载河流区域文件
        self.river_area_file = config.get("river_area_file", "river_area_data.txt")
        self.river_area = np.loadtxt(self.river_area_file, dtype=bool)  # 假设文件为TXT格式，存储河流区域数组

    def display_settings(self):
        print(f"Grid Size: {self.grid_size_x}x{self.grid_size_y}")
        print(f"Grid Width X: {self.grid_width_x}, Grid Width Y: {self.grid_width_y}")
        print(f"Time Step: {self.time_step}")
        print(f"Gravity: {self.gravity}")
        print(f"Erosion Constant: {self.erosion_constant}")
        print(f"Deposition Constant: {self.deposition_constant}")
        print(f"Evaporation Constant: {self.evaporation_constant}")
        print(f"Rainfall Rate: {self.rainfall_rate}")
        print(f"Rainfall Probability: {self.rainfall_probability}")
        print(f"River Speed: {self.river_speed}")
        print(f"Min Pipe Area: {self.min_pipe_area}, Max Pipe Area: {self.max_pipe_area}")
        print(f"Initial Fluid Height: {self.initial_fluid_height}")
        print(f"Terrain File: {self.terrain_file}")
        print(f"Boundary Conditions: {self.boundary_conditions}")

if __name__ == "__main__":
    settings = SimulationSettings(config_file="config.json")
    settings.display_settings()