# solver.py

import numpy as np


# 整个文件都需要注意，在python中，数组第1个是行，第2个是列；因此对于网格(i,j)来说，其左边是(i,j-1)，右边是(i,j+1)，上边是(i-1,j)，下边是(i+1,j)
# 在使用x、y时，x代表横向，y代表纵向
# 在使用i，j时，i代表纵向，j代表横向，i+1代表向下，i-1代表向上，j+1代表向右，j-1代表向左

# 第一步：创建一个函数，用于更新水深数组，考虑降雨和河流的影响。
def apply_rain_and_river(water_height, rainfall_intensity_mm_per_hour, rain_probability,
                         river_speed, river_area, grid_size_x, grid_size_y, time_step):
    """
    更新水深，考虑单步长内的降雨和河流两种情况。

    参数:
    - water_height: 当前水深数组
    - rainfall_intensity_per_second: 每秒的降雨强度 (单位: mm/s)
    - rain_probability: 降雨发生的概率 (值在 0 到 1 之间)
    - river_speed: 河流流速 (单位: m/s)
    - river_area: 河流影响的区域 (布尔数组，形状: (grid_size_x, grid_size_y))
    - grid_size_x, grid_size_y: 网格的尺寸
    - time_step: 时间步长 (单位: s)

    返回值:
    - 更新后的水深数组
    """

    # 转换降雨强度为米/秒
    rainfall_per_second = rainfall_intensity_mm_per_hour / 1000 / 3600  # 从 mm/hour 转换为 m/s

    # 创建一个降雨掩码，决定哪些格子受到降雨影响
    rain_mask = np.random.rand(grid_size_y, grid_size_x) < rain_probability

    # 计算降雨对水深的贡献 (单位：米)
    water_height += np.where(rain_mask, rainfall_per_second * time_step, 0)

    # 计算河流影响区域的水深变化 (单位：米)
    water_height += np.where(river_area, river_speed * time_step, 0)

    return water_height



# 第二步：创建一个函数，用于模拟水流的流动过程，基于管道模型和水位差计算流量。
# 在整个入参中，涉及到上下左右的，指网格的上下左右；但是需要注意二维数组的特征，在python中，数组第1个是行，第2个是列；因此对于i,j来说，其左边是i,j-1，右边是i,j+1，上边是i-1,j，下边是i+1,j
def simulate_water_flow(terrain_height, water_height, outflow_flux_left, outflow_flux_right,
                        outflow_flux_top, outflow_flux_bottom, grid_size_x, grid_size_y,
                        time_step, gravity, grid_width_x, grid_width_y, min_pipe_area, max_pipe_area):
    """
    模拟水流的流动过程，基于管道模型和水位差计算流量。

    参数:
    - terrain_height: 地形高度数组
    - water_height: 当前水位高度数组
    - outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom: 记录水向相邻网格流动的流量
    - grid_size_x, grid_size_y: 网格尺寸
    - time_step: 时间步长
    - gravity: 重力加速度
    - pipe_length: 虚拟管道长度（网格间距）
    - min_pipe_area, max_pipe_area: 管道截面积的最小值和最大值
    

    返回:
    - 更新后的水位高度数组
    - 更新后的流量数组
    """

    # 创建一个新的水高度数组，用于记录更新后的水位
    new_water_height = np.copy(water_height)

    # TODO: 2. 目前这里没有考虑周围一圈的网格怎么计算，直接是没有参与计算；
    for i in range(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            if water_height[i, j] > 1e-6:  # 只有当有足够的水时才计算流动

                # 计算各个方向的水位差
                # 注意：在python中，数组第1个是行，第2个是列；因此对于i,j来说，其左边是i,j-1，右边是i,j+1，上边是i-1,j，下边是i+1,j
                delta_h_left = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i, j - 1] + water_height[i, j - 1])
                delta_h_right = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i, j + 1] + water_height[i, j + 1])
                delta_h_top = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i - 1, j] + water_height[i - 1, j])
                delta_h_bottom = (terrain_height[i, j] + water_height[i, j]) - (terrain_height[i + 1, j] + water_height[i + 1, j])

                # # 动态计算管道的截面积，限制在最小值和最大值之间
                # pipe_area_left = np.clip(abs(delta_h_left * grid_width_y), min_pipe_area, max_pipe_area)
                # pipe_area_right = np.clip(abs(delta_h_right * grid_width_y), min_pipe_area, max_pipe_area)
                # pipe_area_top = np.clip(abs(delta_h_top * grid_width_x), min_pipe_area, max_pipe_area)
                # pipe_area_bottom = np.clip(abs(delta_h_bottom * grid_width_x), min_pipe_area, max_pipe_area)


                # 不限制截面积大小
                # 左侧和右侧水流的管道截面积，应该使用 grid_width_y
                pipe_area_left = abs(delta_h_left * grid_width_y)
                pipe_area_right = abs(delta_h_right * grid_width_y)

                # 顶部和底部水流的管道截面积，应该使用 grid_width_x
                pipe_area_top = abs(delta_h_top * grid_width_x)
                pipe_area_bottom = abs(delta_h_bottom * grid_width_x)

                # 计算水流向各个方向的流量
                # 这是一个递归计算，上一步的通量会影响下一步的通量
                outflow_flux_left[i, j] = max(0, outflow_flux_left[i, j] + time_step * pipe_area_left * gravity * delta_h_left / grid_width_x)
                outflow_flux_right[i, j] = max(0, outflow_flux_right[i, j] + time_step * pipe_area_right * gravity * delta_h_right / grid_width_x)
                outflow_flux_top[i, j] = max(0, outflow_flux_top[i, j] + time_step * pipe_area_top * gravity * delta_h_top / grid_width_y)
                outflow_flux_bottom[i, j] = max(0, outflow_flux_bottom[i, j] + time_step * pipe_area_bottom * gravity * delta_h_bottom / grid_width_y)

                # 防止流量导致水位变为负值 (流量约束)
                total_outflow = outflow_flux_left[i, j] + outflow_flux_right[i, j] + outflow_flux_top[i, j] + outflow_flux_bottom[i, j]
                if total_outflow > water_height[i, j] * grid_width_x * grid_width_y:
                    scaling_factor = water_height[i, j] * grid_width_x * grid_width_y / total_outflow
                    outflow_flux_left[i, j] *= scaling_factor
                    outflow_flux_right[i, j] *= scaling_factor
                    outflow_flux_top[i, j] *= scaling_factor
                    outflow_flux_bottom[i, j] *= scaling_factor

    # 更新水位高度
    for i in range(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            # 计算流入和流出水量
            # 注意：在python中，数组第1个是行，第2个是列；因此对于i,j来说，其左边是i,j-1，右边是i,j+1，上边是i-1,j，下边是i+1,j
            inflow = outflow_flux_right[i, j - 1] + outflow_flux_left[i, j + 1] + outflow_flux_bottom[i - 1, j] + outflow_flux_top[i + 1, j]
            outflow = outflow_flux_left[i, j] + outflow_flux_right[i, j] + outflow_flux_top[i, j] + outflow_flux_bottom[i, j]

            # 更新水位，基于流入和流出计算水的体积变化
            delta_water_volume = time_step * (inflow - outflow)
            new_water_height[i, j] += delta_water_volume / (grid_width_x * grid_width_y)

    # 计算速度
    velocity_x = np.zeros((grid_size_y, grid_size_x))
    velocity_y = np.zeros((grid_size_y, grid_size_x))

    avg_water_depth = (water_height + new_water_height) / 2

    # TODO: 3. 目前这里没有考虑周围一圈的网格怎么计算，直接是没有参与计算；
    for i in range(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):

            # 计算X方向的水流量变化 ΔW_X
            delta_WX = (outflow_flux_right[i, j - 1] - outflow_flux_left[i, j] + outflow_flux_right[i, j] - outflow_flux_left[i, j + 1]) / 2
            
            # 计算Y方向的水流量变化 ΔW_Y
            delta_WY = (outflow_flux_bottom[i - 1, j] - outflow_flux_top[i, j] + outflow_flux_bottom[i, j] - outflow_flux_top[i + 1, j]) / 2

            # X方向速度：l_Y * avg_water_depth * velocity_x = delta_WX
            if avg_water_depth[i, j] > 1e-6:  # 防止除零错误
                velocity_x[i, j] = delta_WX / (grid_width_y * avg_water_depth[i, j])
            else:
                velocity_x[i, j] = 0  # 如果水深非常小，速度设为0

            # Y方向速度：l_X * avg_water_depth * velocity_y = delta_WY
            if avg_water_depth[i, j] > 1e-6:  # 防止除零错误
                velocity_y[i, j] = delta_WY / (grid_width_x * avg_water_depth[i, j])
            else:
                velocity_y[i, j] = 0  # 如果水深非常小，速度设为0

    return new_water_height, velocity_x, velocity_y, outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom

# 第3步：创建一个函数，用于更新地形高度数组，考虑侵蚀和沉积过程。
def calculate_erosion_deposition(terrain_height, grid_size_x, grid_size_y, grid_width_x, grid_width_y, velocity_x, velocity_y, sediment, K_capacity, K_erosion, K_deposition):
    """
    计算每个网格的坡度，分别计算与相邻网格的坡度后求平均。

    参数:
    - terrain_height: 地形高度数组
    - grid_size_x, grid_size_y: 网格尺寸


    返回:
    - erosion_deposition: 每个网格单元的侵蚀或沉积量数组
    """
    
    # 计算sediment transport capacity
    # 这里是否要把一个网格拆成两半，然后分别计算？需要再考虑
    # TODO：这里没有考虑周围一圈的网格怎么计算，直接是没有参与计算；
    erosion_deposition = np.zeros((grid_size_y, grid_size_x))
    
    for i in range(1, grid_size_y - 1):
        for j in range(1, grid_size_x - 1):
            
            # X方向坡度的拆分计算
            delta_z_R = terrain_height[i, j+1] - terrain_height[i, j]
            delta_z_L = terrain_height[i, j] - terrain_height[i, j-1]
            
            L_x_R = np.sqrt(grid_width_x**2 + delta_z_R**2)
            L_x_L = np.sqrt(grid_width_x**2 + delta_z_L**2)
            
            sin_alpha_x_R = delta_z_R / L_x_R
            sin_alpha_x_L = delta_z_L / L_x_L

            # Y方向坡度的拆分计算
            delta_z_B = terrain_height[i+1, j] - terrain_height[i, j]
            delta_z_T = terrain_height[i, j] - terrain_height[i-1, j]
            
            L_y_B = np.sqrt(grid_width_y**2 + delta_z_B**2)
            L_y_T = np.sqrt(grid_width_y**2 + delta_z_T**2)
            
            sin_alpha_y_B = delta_z_B / L_y_B
            sin_alpha_y_T = delta_z_T / L_y_T
            
            # 计算sediment transport capacity，水流的泥沙运输能力
            C_x = K_capacity * (np.abs(sin_alpha_x_R) + np.abs(sin_alpha_x_L)) * abs(velocity_x[i,j]) / 2
            C_y = K_capacity * (np.abs(sin_alpha_y_B) + np.abs(sin_alpha_y_T)) * abs(velocity_y[i,j]) / 2


            # Capacity_sediment_transport值计算
            Capacity = np.sqrt(C_x**2 + C_y**2)
            
            # 侵蚀或沉积计算
            if Capacity > sediment[i, j]:
                # 发生侵蚀
                erosion_deposition[i, j] = K_erosion * (Capacity - sediment[i, j])
                terrain_height[i, j] -= erosion_deposition[i, j] / (grid_width_x * grid_width_y)    # 地形高度降低
                sediment[i, j] += erosion_deposition[i, j]  # 泥沙量增加
            else:
                # 发生沉积
                erosion_deposition[i, j] = K_deposition * (sediment[i, j] - Capacity)
                terrain_height[i, j] += erosion_deposition[i, j] / (grid_width_x * grid_width_y)  # 地形高度升高
                sediment[i, j] -= erosion_deposition[i, j]  # 泥沙量减少

            
    return terrain_height, sediment, erosion_deposition

# 第4步：创建一个函数，用于计算sediment移动
# TODO: 这部分的正确性不确定，需要再检查

# 代码解释：
# 	1.	bilinear_interpolation 函数：
# 	•	用于实现双线性插值。它接收泥沙浓度数组 sediment 和非整数坐标  (x, y) ，通过相邻四个网格点的泥沙浓度进行加权平均，计算出插值位置的泥沙浓度。
# 	•	边界处理确保回溯的坐标不会超出网格范围。
# 	2.	semi_lagrangian_advection 函数：
# 	•	使用半拉格朗日方法计算泥沙的对流。
# 	•	对于每个网格点，根据水流速度  \vec{v} = (u, v) ，逆向回溯时间步  \Delta t ，计算粒子在上一个时刻的位置。
# 	•	通过插值计算出上一个时刻的位置的泥沙浓度，并更新当前网格点的泥沙浓度。
def bilinear_interpolation(sediment, x, y, grid_size_x, grid_size_y):
    """
    双线性插值，用于估算非整数位置的泥沙浓度。
    
    参数:
    - sediment: 泥沙浓度数组
    - x, y: 非整数位置
    - grid_size_x, grid_size_y: 网格尺寸
    
    返回:
    - 插值后的泥沙浓度
    """
    # 获取整数坐标
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    # 边界处理，确保坐标不超出网格范围
    x0 = np.clip(x0, 0, grid_size_x - 1)
    x1 = np.clip(x1, 0, grid_size_x - 1)
    y0 = np.clip(y0, 0, grid_size_y - 1)
    y1 = np.clip(y1, 0, grid_size_y - 1)

    # 插值权重
    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1

    # 双线性插值公式
    return (sediment[y0, x0] * wx0 * wy0 +
            sediment[y1, x0] * wx1 * wy0 +
            sediment[y0, x1] * wx0 * wy1 +
            sediment[y1, x1] * wx1 * wy1)

def semi_lagrangian_advection(sediment, velocity_x, velocity_y, grid_size_x, grid_size_y, time_step):
    """
    使用半拉格朗日方法计算泥沙对流。
    
    参数:
    - sediment: 泥沙浓度数组
    - velocity_x, velocity_y: 水流在X和Y方向的速度数组
    - grid_size_x, grid_size_y: 网格尺寸
    - delta_t: 时间步长
    
    返回:
    - updated_sediment: 更新后的泥沙浓度数组
    """
    # 初始化新的泥沙浓度数组
    updated_sediment = np.zeros_like(sediment)

    # 遍历每个网格点
    for i in range(grid_size_y):  # 遍历行方向，即Y方向
        for j in range(grid_size_x):  # 遍历列方向，即X方向
            # 当前的水流速度
            u = velocity_x[i, j]  # x 方向的速度
            v = velocity_y[i, j]  # y 方向的速度

            # 逆向回溯，计算上一个时刻的位置
            x_prev = j - u * time_step  # 注意：j 对应物理上的 x 方向
            y_prev = i - v * time_step  # 注意：i 对应物理上的 y 方向

            # 使用双线性插值计算回溯位置的泥沙浓度
            updated_sediment[i, j] = bilinear_interpolation(sediment, x_prev, y_prev, grid_size_x, grid_size_y)

    return updated_sediment

# 第5步：创建一个函数，用于计算水的蒸发
def evaporation(water_height, grid_size_x, grid_size_y, K_evaporation, time_step):
    """
    模拟水的蒸发过程。
    
    参数:
    - water_height: 当前网格的水深数组
    - grid_size_x, grid_size_y: 网格的尺寸
    - K_evaporation: 蒸发常数
    - time_step: 时间步长
    
    返回:
    - updated_water_height: 更新后的水深数组
    """
    # 初始化新的水深数组
    updated_water_height = np.copy(water_height)

    # 遍历每个网格点，计算新的水深
    for i in range(grid_size_y):
        for j in range(grid_size_x):
            # 根据蒸发公式计算更新后的水深
            updated_water_height[i, j] = water_height[i, j] * (1 - K_evaporation * time_step)

            # 确保水深不为负值
            if updated_water_height[i, j] < 0:
                updated_water_height[i, j] = 0

    return updated_water_height


# 构建一个step函数，用于执行整个模拟过程
def step(settings, water_height, terrain_height, outflow_flux_left, outflow_flux_right,
         outflow_flux_top, outflow_flux_bottom, velocity_x, velocity_y,
         sediment):
    """
    执行整个模拟过程的一个时间步。

    参数:
    - settings: SimulationSettings对象, 包含所有的配置参数
    - water_height: 当前水深数组
    - terrain_height: 地形高度数组
    - outflow_flux_left, outflow_flux_right, outflow_flux_top, outflow_flux_bottom: 水流方向上的通量数组
    - velocity_x, velocity_y: 水流在X和Y方向的速度
    - sediment: 泥沙数组

    返回:
    - water_height: 更新后的水深数组
    - terrain_height: 更新后的地形高度数组
    - sediment: 更新后的泥沙数组
    - velocity_x: 更新后的X方向速度
    - velocity_y: 更新后的Y方向速度
    """
    
    # 第1步: 应用降雨和河流的影响
    water_height = apply_rain_and_river(water_height, settings.rainfall_intensity_mm_per_hour, settings.rainfall_probability,
                                        settings.river_speed, settings.river_area, settings.grid_size_x, settings.grid_size_y, 
                                        settings.time_step)
    
    # 第2步: 模拟水流流动
    water_height, velocity_x, velocity_y, outflow_flux_left, outflow_flux_right, \
    outflow_flux_top, outflow_flux_bottom = simulate_water_flow(
        terrain_height, water_height, outflow_flux_left, outflow_flux_right,
        outflow_flux_top, outflow_flux_bottom, settings.grid_size_x, settings.grid_size_y,
        settings.time_step, settings.gravity, settings.grid_width_x, settings.grid_width_y, 
        settings.min_pipe_area, settings.max_pipe_area)
    
    # 第3步: 计算侵蚀与沉积过程
    terrain_height, sediment, erosion_deposition = calculate_erosion_deposition(terrain_height, settings.grid_size_x, settings.grid_size_y,
                                                            settings.grid_width_x, settings.grid_width_y, 
                                                            velocity_x, velocity_y, sediment, settings.erosion_constant, 
                                                            settings.erosion_constant, settings.deposition_constant)
    
    # 第4步: 泥沙对流计算
    sediment = semi_lagrangian_advection(sediment, velocity_x, velocity_y, settings.grid_size_x, settings.grid_size_y, settings.time_step)
    
    # 第5步: 计算水的蒸发
    water_height = evaporation(water_height, settings.grid_size_x, settings.grid_size_y, settings.evaporation_constant, settings.time_step)
    
    # 返回更新后的水深、地形高度和泥沙数组
    return water_height, terrain_height, sediment, velocity_x, velocity_y, erosion_deposition