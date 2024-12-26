# 唐家山堰塞湖溃坝模拟系统

这是一个基于Python的水动力学模拟系统，用于模拟堰塞湖溃坝过程中的水流运动、泥沙输运和地形演变。

## 功能特点

- 水流运动模拟：基于管道模型的水流计算
- 泥沙输运模拟：包括侵蚀和沉积过程
- 地形演变：实时更新地形高程
- 可视化功能：支持2D和3D实时可视化
- 高性能计算：支持多进程并行计算和Numba JIT加速

## 系统要求

- Python 3.8+
- 支持多核处理器（推荐）
- 足够的内存来处理大规模网格计算

## 安装方法

1. 克隆仓库：
```bash
git clone [repository_url]
cd TangDammedLake
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备输入数据：
   - 地形数据文件（terrain_data.txt）
   - 河流区域数据文件（river_area_data.txt）
   - 初始水深数据文件（initial_water_height.txt）
   - 配置文件（config.json）

2. 配置参数：
   编辑 `config.json` 文件，设置模拟参数：
   ```json
   {
     "time_step": 0.01,
     "gravity": 9.81,
     "grid_width_x": 1.0,
     "grid_width_y": 1.0,
     "rainfall_intensity_mm_per_hour": 0.02,
     "rainfall_probability": 0.1,
     "river_speed": 1.0,
     ...
   }
   ```

3. 运行模拟：
```bash
python main.py
```

## 可视化选项

系统提供三种可视化模式：
1. 无可视化（仅计算）
2. 2D实时可视化
3. 3D实时可视化

可以在 `main.py` 中通过修改 `VisualizationMode` 参数来选择：
```python
final_state = run_simulation(settings, initial_state, VisualizationMode.MODE_2D, update_interval=1)
```

## 性能优化

系统提供了三个版本的求解器：
1. `solver.py`: 基础版本
2. `solver_optimized.py`: Numba优化版本
3. `solver_multiprocess.py`: 多进程并行计算版本

可以根据需要在 `main.py` 中选择不同的求解器：
```python
import solver_multiprocess as solver  # 使用多进程版本
```

## 输出结果

模拟结果包括：
- 水深分布
- 流速场
- 地形变化
- 泥沙输运
- 侵蚀/沉积分布

## 注意事项

1. 确保输入数据文件格式正确
2. 多进程版本需要足够的系统内存
3. 可视化可能会影响计算性能
4. 建议先使用小规模网格测试

## 许可证

[添加许可证信息]

## 贡献者

[添加贡献者信息] 