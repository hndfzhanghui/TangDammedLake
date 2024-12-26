import numpy as np
from numba import jit, njit, prange

@njit
def apply_rain_and_river(water_height, rainfall_intensity_mm_per_hour, rain_probability,
                         river_speed, river_area, grid_size_x, grid_size_y, time_step):
    rainfall_per_second = rainfall_intensity_mm_per_hour / 1000 / 3600
    rain_mask = np.random.rand(grid_size_y, grid_size_x) < rain_probability
    water_height += np.where(rain_mask, rainfall_per_second * time_step, 0)
    water_height += np.where(river_area, river_speed * time_step, 0)
    return water_height

@njit(parallel=True)
def simulate_water_flow(terrain_height, water_height, outflow_flux_left, outflow_flux_right,
                        outflow_flux_top, outflow_flux_bottom, grid_size_x, grid_size_y,
                        time_step, gravity, grid_width_x, grid_width_y, min_pipe_area, max_pipe_area):
    new_water_height = np.copy(water_height)
    velocity_x = np.zeros((grid_size_y, grid_size_x))
    velocity_y = np.zeros((grid_size_y, grid_size_x))

    # 第一阶段：计算流量
    for i in prange(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            if water_height[i, j] > 1e-3:
                delta_h_left = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i, j - 1] + water_height[i, j - 1])
                delta_h_right = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i, j + 1] + water_height[i, j + 1])
                delta_h_top = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i - 1, j] + water_height[i - 1, j])
                delta_h_bottom = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i + 1, j] + water_height[i + 1, j])

                pipe_area_left = abs(delta_h_left * grid_width_y)
                pipe_area_right = abs(delta_h_right * grid_width_y)
                pipe_area_top = abs(delta_h_top * grid_width_x)
                pipe_area_bottom = abs(delta_h_bottom * grid_width_x)

                outflow_flux_left[i, j] = max(0, outflow_flux_left[i, j] + time_step * pipe_area_left * gravity * delta_h_left / grid_width_x)
                outflow_flux_right[i, j] = max(0, outflow_flux_right[i, j] + time_step * pipe_area_right * gravity * delta_h_right / grid_width_x)
                outflow_flux_top[i, j] = max(0, outflow_flux_top[i, j] + time_step * pipe_area_top * gravity * delta_h_top / grid_width_y)
                outflow_flux_bottom[i, j] = max(0, outflow_flux_bottom[i, j] + time_step * pipe_area_bottom * gravity * delta_h_bottom / grid_width_y)

                total_outflow = outflow_flux_left[i, j] + outflow_flux_right[i, j] + outflow_flux_top[i, j] + outflow_flux_bottom[i, j]
                if total_outflow > water_height[i, j] * grid_width_x * grid_width_y:
                    scaling_factor = water_height[i, j] * grid_width_x * grid_width_y / total_outflow
                    outflow_flux_left[i, j] *= scaling_factor
                    outflow_flux_right[i, j] *= scaling_factor
                    outflow_flux_top[i, j] *= scaling_factor
                    outflow_flux_bottom[i, j] *= scaling_factor

    # 第二阶段：更新水位
    for i in prange(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            inflow = outflow_flux_right[i, j - 1] + outflow_flux_left[i, j + 1] + outflow_flux_bottom[i - 1, j] + outflow_flux_top[i + 1, j]
            outflow = outflow_flux_left[i, j] + outflow_flux_right[i, j] + outflow_flux_top[i, j] + outflow_flux_bottom[i, j]
            delta_water_volume = time_step * (inflow - outflow)
            new_water_height[i, j] += delta_water_volume / (grid_width_x * grid_width_y)

    avg_water_depth = (water_height + new_water_height) / 2

    # 第三阶段：计算速度
    for i in prange(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            delta_WX = (outflow_flux_right[i, j - 1] - outflow_flux_left[i, j] + outflow_flux_right[i, j] - outflow_flux_left[i, j + 1]) / 2
            delta_WY = (outflow_flux_bottom[i - 1, j] - outflow_flux_top[i, j] + outflow_flux_bottom[i, j] - outflow_flux_top[i + 1, j]) / 2

            if avg_water_depth[i, j] > 1e-6:
                velocity_x[i, j] = delta_WX / (grid_width_y * avg_water_depth[i, j])
                velocity_y[i, j] = delta_WY / (grid_width_x * avg_water_depth[i, j])

    return new_water_height, velocity_x, velocity_y, outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom

@njit(parallel=True)
def calculate_erosion_deposition(terrain_height, grid_size_x, grid_size_y, grid_width_x, grid_width_y, 
                               velocity_x, velocity_y, sediment, K_capacity, K_erosion, K_deposition):
    erosion_deposition = np.zeros((grid_size_y, grid_size_x))
    
    for i in prange(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            delta_z_R = terrain_height[i, j+1] - terrain_height[i, j]
            delta_z_L = terrain_height[i, j] - terrain_height[i, j-1]
            
            L_x_R = np.sqrt(grid_width_x**2 + delta_z_R**2)
            L_x_L = np.sqrt(grid_width_x**2 + delta_z_L**2)
            
            sin_alpha_x_R = delta_z_R / L_x_R
            sin_alpha_x_L = delta_z_L / L_x_L

            delta_z_B = terrain_height[i+1, j] - terrain_height[i, j]
            delta_z_T = terrain_height[i, j] - terrain_height[i-1, j]
            
            L_y_B = np.sqrt(grid_width_y**2 + delta_z_B**2)
            L_y_T = np.sqrt(grid_width_y**2 + delta_z_T**2)
            
            sin_alpha_y_B = delta_z_B / L_y_B
            sin_alpha_y_T = delta_z_T / L_y_T
            
            C_x = K_capacity * (np.abs(sin_alpha_x_R) + np.abs(sin_alpha_x_L)) * abs(velocity_x[i,j]) / 2
            C_y = K_capacity * (np.abs(sin_alpha_y_B) + np.abs(sin_alpha_y_T)) * abs(velocity_y[i,j]) / 2

            Capacity = np.sqrt(C_x**2 + C_y**2)
            
            if Capacity > sediment[i, j]:
                erosion_deposition[i, j] = K_erosion * (Capacity - sediment[i, j])
                terrain_height[i, j] -= erosion_deposition[i, j] / (grid_width_x * grid_width_y)
                sediment[i, j] += erosion_deposition[i, j]
            else:
                erosion_deposition[i, j] = K_deposition * (sediment[i, j] - Capacity)
                terrain_height[i, j] += erosion_deposition[i, j] / (grid_width_x * grid_width_y)
                sediment[i, j] -= erosion_deposition[i, j]
            
    return terrain_height, sediment, erosion_deposition

@njit
def bilinear_interpolation(sediment, x, y, grid_size_x, grid_size_y):
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    
    # 确保坐标在网格范围内
    x0 = max(0, min(x0, grid_size_x - 2))
    x1 = max(1, min(x1, grid_size_x - 1))
    y0 = max(0, min(y0, grid_size_y - 2))
    y1 = max(1, min(y1, grid_size_y - 1))
    
    # 计算权重
    wx = x - x0
    wy = y - y0
    
    # 双线性插值
    return (sediment[y0, x0] * (1 - wx) * (1 - wy) +
            sediment[y0, x1] * wx * (1 - wy) +
            sediment[y1, x0] * (1 - wx) * wy +
            sediment[y1, x1] * wx * wy)

@njit(parallel=True)
def semi_lagrangian_advection(sediment, velocity_x, velocity_y, grid_size_x, grid_size_y, time_step):
    new_sediment = np.zeros_like(sediment)
    
    for i in prange(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            # 回溯计算
            x = j - velocity_x[i, j] * time_step
            y = i - velocity_y[i, j] * time_step
            
            # 确保坐标在网格范围内
            x = max(0, min(x, grid_size_x - 1))
            y = max(0, min(y, grid_size_y - 1))
            
            # 使用双线性插值计算泥沙浓度
            new_sediment[i, j] = bilinear_interpolation(sediment, x, y, grid_size_x, grid_size_y)
    
    return new_sediment

@njit
def evaporation(water_height, grid_size_x, grid_size_y, K_evaporation, time_step):
    """
    计算水面蒸发。
    """
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            if water_height[i, j] > 0:
                evap_amount = K_evaporation * time_step
                water_height[i, j] = max(0, water_height[i, j] - evap_amount)
    return water_height

def step(settings, water_height, terrain_height, outflow_flux_left, outflow_flux_right,
         outflow_flux_top, outflow_flux_bottom, velocity_x, velocity_y,
         sediment):
    """
    执行一个时间步长的模拟。
    """
    # 1. 更新水深，考虑降雨和河流
    water_height = apply_rain_and_river(
        water_height, settings.rainfall_intensity_mm_per_hour, settings.rainfall_probability,
        settings.river_speed, settings.river_area,
        settings.grid_size_x, settings.grid_size_y, settings.time_step
    )
    
    # 2. 模拟水流运动
    water_height, velocity_x, velocity_y, outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom = simulate_water_flow(
        terrain_height, water_height,
        outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom,
        settings.grid_size_x, settings.grid_size_y,
        settings.time_step, settings.gravity,
        settings.grid_width_x, settings.grid_width_y,
        settings.min_pipe_area, settings.max_pipe_area
    )
    
    # 3. 计算侵蚀和沉积
    terrain_height, sediment, erosion_deposition = calculate_erosion_deposition(
        terrain_height,
        settings.grid_size_x, settings.grid_size_y,
        settings.grid_width_x, settings.grid_width_y,
        velocity_x, velocity_y, sediment,
        settings.erosion_constant, settings.erosion_constant, settings.deposition_constant
    )
    
    # 4. 计算泥沙运移
    sediment = semi_lagrangian_advection(
        sediment, velocity_x, velocity_y,
        settings.grid_size_x, settings.grid_size_y,
        settings.time_step
    )
    
    # # 5. 计算蒸发
    # water_height = evaporation(
    #     water_height,
    #     settings.grid_size_x, settings.grid_size_y,
    #     settings.evaporation_constant, settings.time_step
    # )
    
    return (water_height, terrain_height, sediment, velocity_x, velocity_y, erosion_deposition) 