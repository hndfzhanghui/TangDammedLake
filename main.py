import numpy as np
from enum import Enum
from settings import SimulationSettings
from animation import Visualizer3D
from animation_2d import Visualizer2D

class VisualizationMode(Enum):
    NONE = 0    # 不显示可视化
    MODE_2D = 1 # 2D可视化
    MODE_3D = 2 # 3D可视化

def run_simulation(settings, initial_state, vis_mode=VisualizationMode.NONE, update_interval=1):
    """
    运行模拟计算
    
    Args:
        settings: SimulationSettings对象，包含配置参数
        initial_state: 初始状态字典
        vis_mode: 可视化模式，默认不显示
        update_interval: 每隔多少步更新一次可视化，默认为1
    """
    # 初始化状态
    state = {k: v.copy() for k, v in initial_state.items()}
    
    # 初始化可视化器
    visualizer = None
    if vis_mode == VisualizationMode.MODE_2D:
        visualizer = Visualizer2D()
    elif vis_mode == VisualizationMode.MODE_3D:
        visualizer = Visualizer3D()
    
    if visualizer:
        visualizer.initialize(settings, state)

    # 运行模拟
    num_steps = 3600
    for step in range(num_steps):
        # 执行计算
        from solver import step as compute_step
        state['water_height'], state['terrain_height'], state['sediment'], \
        state['velocity_x'], state['velocity_y'], state['erosion_deposition'] = compute_step(
            settings,
            state['water_height'],
            state['terrain_height'],
            state['outflow_flux_left'],
            state['outflow_flux_right'],
            state['outflow_flux_top'],
            state['outflow_flux_bottom'],
            state['velocity_x'],
            state['velocity_y'],
            state['sediment']
        )
        
        # 如果需要可视化且到达更新间隔，则更新显示
        if visualizer and step % update_interval == 0:
            visualizer.update(step + 1, state)
    
    # 如果有可视化，等待窗口关闭
    if visualizer:
        visualizer.show()
    
    return state

def main():
    # 加载配置
    settings = SimulationSettings("config.json")

    # 初始化状态
    initial_state = {
        'water_height': np.full((settings.grid_size_y, settings.grid_size_x), settings.initial_fluid_height),
        'terrain_height': settings.terrain_height.copy(),
        'sediment': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'velocity_x': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'velocity_y': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'outflow_flux_left': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'outflow_flux_right': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'outflow_flux_top': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'outflow_flux_bottom': np.zeros((settings.grid_size_y, settings.grid_size_x)),
        'erosion_deposition': np.zeros((settings.grid_size_y, settings.grid_size_x))
    }

    # 运行模拟 - 可以选择不同的可视化模式
    # final_state = run_simulation(settings, initial_state)  # 不显示可视化
    final_state = run_simulation(settings, initial_state, VisualizationMode.MODE_3D, update_interval=5)  # 2D可视化，每5步更新一次
    # final_state = run_simulation(settings, initial_state, VisualizationMode.MODE_3D, update_interval=1)  # 3D可视化，每步都更新

    # 输出结果
    print("\n模拟完成！")
    print("最终地形高度范围:", np.min(final_state['terrain_height']), "到", np.max(final_state['terrain_height']))
    print("最终水深范围:", np.min(final_state['water_height']), "到", np.max(final_state['water_height']))

if __name__ == "__main__":
    main()
