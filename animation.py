import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from visualizer import Visualizer

# 设置中文字体
try:
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # macOS
except:
    try:
        font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')  # Linux
    except:
        try:
            font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # Windows
        except:
            print("无法找到合适的中文字体，将使用系统默认字体。")
            font = FontProperties()

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Visualizer3D(Visualizer):
    def __init__(self):
        self.fig = None
        self.axs = None
        self.settings = None
        self.initial_terrain_height = None
        self.paused = False
        self.cumulative_erosion = None
        
        # 图形元素
        self.terrain_surface = None
        self.water_surface = None
        self.erosion_plot = None
        self.quiver = None
        self.sediment_plot = None
        self.button = None
        
        # 网格数据
        self.X = None
        self.Y = None
        
    def create_terrain_water_surface(self, ax, terrain_height, water_height):
        """创建地形和水面的mesh表面"""
        # 绘制地形
        terrain_surface = ax.plot_surface(self.X, self.Y, terrain_height,
                                        cmap='terrain',
                                        alpha=0.7)
        
        # 绘制水面（只在有水的地方）
        water_mask = water_height > 0
        if np.any(water_mask):
            # 创建带有mask的水面高度数组
            water_surface_height = np.ma.masked_array(
                terrain_height + water_height,
                mask=~water_mask
            )
            water_surface = ax.plot_surface(self.X, self.Y, water_surface_height,
                                          cmap='Blues',
                                          alpha=0.4)
            return terrain_surface, water_surface
        
        return terrain_surface, None
        
    def initialize(self, settings, initial_state):
        """初始化3D可视化器"""
        self.settings = settings
        self.initial_terrain_height = initial_state['terrain_height'].copy()
        self.cumulative_erosion = np.zeros_like(initial_state['terrain_height'])
        
        # 创建图形和子图
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('水力侵蚀模拟', fontproperties=font)

        # 创建3D子图
        self.axs = [
            self.fig.add_subplot(221, projection='3d'),  # 地形与水深
            self.fig.add_subplot(222, projection='3d'),  # 累积搬移量
            self.fig.add_subplot(223),                   # 流速流向
            self.fig.add_subplot(224, projection='3d')   # 实时搬移量
        ]

        # 准备3D网格数据
        x = np.arange(initial_state['terrain_height'].shape[1])
        y = np.arange(initial_state['terrain_height'].shape[0])
        self.X, self.Y = np.meshgrid(x, y)

        # 设置视角和高度范围
        for ax in [self.axs[0], self.axs[1], self.axs[3]]:
            ax.view_init(elev=30, azim=225)  # 修改方位角到225度来反转x方向

        # 设置第一个图的高度范围
        self.axs[0].set_zlim(0, 80)

        # 初始化3D图形
        self.terrain_surface, self.water_surface = self.create_terrain_water_surface(
            self.axs[0], initial_state['terrain_height'], initial_state['water_height']
        )
        
        # 初始化累积搬移量图，设置自适应高度范围
        self.erosion_plot = self.axs[1].plot_surface(self.X, self.Y, self.cumulative_erosion, cmap='RdBu_r')
        
        # 初始化流速矢量图，设置固定的坐标范围
        max_velocity = np.sqrt(np.max(initial_state['velocity_x']**2 + initial_state['velocity_y']**2))
        self.quiver = self.axs[2].quiver(initial_state['velocity_x'], initial_state['velocity_y'])
        self.axs[2].set_xlim(0, self.settings.grid_size_x)
        self.axs[2].set_ylim(0, self.settings.grid_size_y)
        
        # 初始化实时搬移量图，设置自适应高度范围
        self.sediment_plot = self.axs[3].plot_surface(self.X, self.Y, initial_state['erosion_deposition'], cmap='RdBu_r')

        # 设置子图标题
        self.axs[0].set_title('地形与水深', fontproperties=font)
        self.axs[1].set_title('累积搬移量', fontproperties=font)
        self.axs[2].set_title('流速流向', fontproperties=font)
        self.axs[3].set_title('实时搬移量', fontproperties=font)

        # 创建暂停/播放按钮
        ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.button = Button(ax_button, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')
        self.button.on_clicked(self.pause_play)

        plt.tight_layout()
        
    def update(self, step, state):
        """更新3D可视化"""
        if self.paused:
            return

        # 更新累积搬移量
        self.cumulative_erosion += state['erosion_deposition']

        # 清除旧的3D图形
        for ax in [self.axs[0], self.axs[1], self.axs[3]]:
            ax.clear()

        # 重新设置视角
        for ax in [self.axs[0], self.axs[1], self.axs[3]]:
            ax.view_init(elev=30, azim=225)  # 修改方位角到225度来反转x方向
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('高度 (m)')

        # 设置第一个图的固定高度范围
        self.axs[0].set_zlim(0, 80)

        # 更新地形和水面
        self.terrain_surface, self.water_surface = self.create_terrain_water_surface(
            self.axs[0], state['terrain_height'], state['water_height']
        )
        self.axs[0].set_title('地形与水深', fontproperties=font)

        # 更新累积搬移量图（自适应高度范围）
        self.axs[1].plot_surface(self.X, self.Y, self.cumulative_erosion, cmap='RdBu_r')
        self.axs[1].set_title('累积搬移量', fontproperties=font)

        # 更新流速矢量图
        self.axs[2].clear()
        self.axs[2].set_title('流速流向', fontproperties=font)
        
        # 计算速度场
        magnitude = np.sqrt(state['velocity_x'] ** 2 + state['velocity_y'] ** 2)
        mask = magnitude > 1e-6
        
        if np.any(mask):
            self.axs[2].quiver(self.X[mask], self.Y[mask],
                             state['velocity_x'][mask],
                             state['velocity_y'][mask])
            
            # 设置固定的坐标范围
            self.axs[2].set_xlim(0, self.settings.grid_size_x)
            self.axs[2].set_ylim(0, self.settings.grid_size_y)

        # 更新实时搬移量图（自适应高度范围）
        self.axs[3].plot_surface(self.X, self.Y, state['erosion_deposition'], cmap='RdBu_r')
        self.axs[3].set_title('实时搬移量', fontproperties=font)

        self.fig.suptitle(f'水力侵蚀模拟 - 步骤 {step}', fontproperties=font)
        plt.pause(0.01)  # 给matplotlib时间更新图形
        
    def pause_play(self, event):
        """暂停/播放切换"""
        self.paused = not self.paused
        self.button.label.set_text('Play' if self.paused else 'Pause')
        
    def show(self):
        """显示图形"""
        plt.show()