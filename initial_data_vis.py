# 初始数据可视化

import matplotlib.pyplot as plt
import numpy as np

# 读取初始数据
initial_water_height = np.loadtxt('initial_water_height.txt')
initial_terrain_height = np.loadtxt('terrain_data.txt')

# 在1个窗口用2个图片可视化地形和水深
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(initial_terrain_height, cmap='terrain', interpolation='bilinear')
plt.colorbar(label='Terrain Height (m)')
plt.title('Initial Terrain Height')
plt.subplot(1, 2, 2)
plt.imshow(initial_water_height, cmap='Blues', interpolation='bilinear')
plt.colorbar(label='Water Height (m)')
plt.title('Initial Water Height')
plt.show()


