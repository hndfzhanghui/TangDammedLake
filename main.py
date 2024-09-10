import numpy as np
from settings import SimulationSettings
from solver import step

def main():
    # 加载配置
    settings = SimulationSettings("config.json")

    # 初始化数组
    water_height = np.full((settings.grid_size_y, settings.grid_size_x), settings.initial_fluid_height)
    terrain_height = settings.terrain_height.copy()
    outflow_flux_left = np.zeros((settings.grid_size_y, settings.grid_size_x))
    outflow_flux_right = np.zeros((settings.grid_size_y, settings.grid_size_x))
    outflow_flux_top = np.zeros((settings.grid_size_y, settings.grid_size_x))
    outflow_flux_bottom = np.zeros((settings.grid_size_y, settings.grid_size_x))
    velocity_x = np.zeros((settings.grid_size_y, settings.grid_size_x))
    velocity_y = np.zeros((settings.grid_size_y, settings.grid_size_x))
    sediment = np.zeros((settings.grid_size_y, settings.grid_size_x))

    # 模拟参数
    total_time = 3600  # 总模拟时间（秒）
    num_steps = int(total_time / settings.time_step)

    # 主模拟循环
    for i in range(num_steps):
        water_height, terrain_height, sediment, velocity_x, velocity_y, erosion_deposition = step(
            settings, water_height, terrain_height, outflow_flux_left, outflow_flux_right,
            outflow_flux_top, outflow_flux_bottom, velocity_x, velocity_y, sediment
        )

        # 每100步打印一次进度
        if i % 100 == 0:
            print(f"完成步骤 {i}/{num_steps}")

    # 模拟结束后打印一些统计信息
    print("\n模拟完成")
    print(f"最终平均水深: {np.mean(water_height):.4f} 米")
    print(f"最终最大水深: {np.max(water_height):.4f} 米")
    print(f"地形高度变化范围: {np.min(terrain_height):.4f} 到 {np.max(terrain_height):.4f} 米")
    print(f"最大沉积物浓度: {np.max(sediment):.4f}")
    print(f"最大流速: {np.max(np.sqrt(velocity_x**2 + velocity_y**2)):.4f} 米/秒")

if __name__ == "__main__":
    main()
