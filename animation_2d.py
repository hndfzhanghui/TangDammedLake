import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button
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

class Visualizer2D(Visualizer):
    def __init__(self):
        self.fig = None
        self.axs = None
        self.settings = None
        self.initial_terrain_height = None
        self.paused = False
        
        # 图形元素
        self.im_terrain_change = None
        self.im_combined = None
        self.im_water = None  # 新增：水深图
        self.quiver = None
        self.im_erosion = None
        self.terrain_change_cbar = None
        self.combined_cbar = None
        self.water_cbar = None  # 新增：水深图的颜色条
        self.erosion_cbar = None
        
    def initialize(self, settings, initial_state):
        """初始化2D可视化器"""
        self.settings = settings
        self.initial_terrain_height = initial_state['terrain_height'].copy()
        
        # 创建图形和子图
        self.fig = plt.figure(figsize=(18, 10))  # 加宽图形以容纳新的子图
        gs = self.fig.add_gridspec(2, 3)  # 2行3列的网格
        self.axs = np.array([
            [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[0, 2])],
            [self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1]), self.fig.add_subplot(gs[1, 2])]
        ])
        self.fig.suptitle('水力侵蚀模拟', fontproperties=font)

        # 初始化子图
        # 1. 地形变化图
        self.im_terrain_change = self.axs[0, 0].imshow(
            np.zeros_like(initial_state['terrain_height']),
            cmap='RdBu_r',
            norm=plt.Normalize(vmin=-1.0, vmax=1.0),
            origin='lower'  # y轴向上为正
        )
        
        # 2. 地形与水深组合图
        self.im_combined = self.axs[0, 1].imshow(
            initial_state['terrain_height'],
            cmap='YlOrBr',
            origin='lower'  # y轴向上为正
        )
        
        # 3. 水深图
        water_mask = initial_state['water_height'] > 0
        masked_water = np.ma.masked_array(
            initial_state['water_height'],
            mask=~water_mask
        )
        self.im_water = self.axs[0, 2].imshow(
            masked_water,
            cmap='Blues',
            norm=plt.Normalize(vmin=0, vmax=np.max(initial_state['water_height']) or 1),
            origin='lower'  # y轴向上为正
        )
        
        # 4. 流速矢量图
        self.quiver = self.axs[1, 0].quiver(
            initial_state['velocity_x'],
            initial_state['velocity_y']
        )
        
        # 5. 侵蚀/沉积图
        self.im_erosion = self.axs[1, 1].imshow(
            initial_state['sediment'],
            cmap='RdBu_r',
            norm=plt.Normalize(vmin=-0.1, vmax=0.1),
            origin='lower'  # y轴向上为正
        )

        # 设置子图标题和颜色条
        self.axs[0, 0].set_title('地形变化', fontproperties=font)
        self.terrain_change_cbar = plt.colorbar(self.im_terrain_change, ax=self.axs[0, 0])
        self.terrain_change_cbar.set_label('高度变化 (m)', fontproperties=font)

        self.axs[0, 1].set_title('地形与水深', fontproperties=font)
        self.combined_cbar = plt.colorbar(self.im_combined, ax=self.axs[0, 1])
        self.combined_cbar.set_label('高度/深度 (m)', fontproperties=font)

        self.axs[0, 2].set_title('水深分布', fontproperties=font)
        self.water_cbar = plt.colorbar(self.im_water, ax=self.axs[0, 2])
        self.water_cbar.set_label('水深 (m)', fontproperties=font)

        self.axs[1, 0].set_title('流速流向', fontproperties=font)

        self.axs[1, 1].set_title('侵蚀/沉积', fontproperties=font)
        self.erosion_cbar = plt.colorbar(self.im_erosion, ax=self.axs[1, 1])
        self.erosion_cbar.set_label('侵蚀/沉积 (m)', fontproperties=font)

        # 隐藏多余的子图
        self.axs[1, 2].set_visible(False)

        # 创建暂停/播放按钮
        ax_button = plt.axes([0.85, 0.05, 0.1, 0.075])  # 调整按钮位置
        self.button = Button(ax_button, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')
        self.button.on_clicked(self.pause_play)

        # 为所有子图设置相同的坐标范围和方向
        for ax in self.axs.flat:
            if ax.get_visible():  # 只设置可见的子图
                ax.set_xlim(0, self.settings.grid_size_x - 1)
                ax.set_ylim(0, self.settings.grid_size_y - 1)  # y轴范围保持一致

        plt.tight_layout()

    def update(self, step, state):
        """更新2D可视化"""
        if self.paused:
            return

        # 更新地形高度图
        terrain_change = state['terrain_height'] - self.initial_terrain_height
        self.im_terrain_change.set_array(terrain_change)
        self.im_terrain_change.set_clim(vmin=-0.05, vmax=0.05)
        self.terrain_change_cbar.update_normal(self.im_terrain_change)

        # 更新地形与水深组合图
        combined_data = state['terrain_height'].copy()
        water_mask = state['water_height'] > 0
        combined_data[water_mask] += state['water_height'][water_mask]
        self.im_combined.set_array(combined_data)
        self.im_combined.set_clim(vmin=np.min(state['terrain_height']), vmax=np.max(combined_data))
        self.combined_cbar.update_normal(self.im_combined)

        # 使用自定义颜色映射
        colors = plt.cm.YlOrBr(plt.Normalize()(state['terrain_height']))
        colors[water_mask] = plt.cm.Blues(plt.Normalize()(state['water_height'][water_mask]))
        self.im_combined.set_data(colors)

        # 更新水深图（新增）
        masked_water = np.ma.masked_array(
            state['water_height'],
            mask=~water_mask
        )
        self.im_water.set_array(masked_water)
        if np.any(water_mask):
            self.im_water.set_clim(vmin=0, vmax=np.max(state['water_height']))
        self.water_cbar.update_normal(self.im_water)

        # 更新流速矢量图
        self.axs[1, 0].clear()
        self.axs[1, 0].set_title('流速流向', fontproperties=font)

        # 生成二维网格
        x = np.arange(state['velocity_x'].shape[1])
        y = np.arange(state['velocity_x'].shape[0])
        X, Y = np.meshgrid(x, y)

        # 设定最小速度阈值，过滤掉接近零的速度
        magnitude = np.sqrt(state['velocity_x'] ** 2 + state['velocity_y'] ** 2)
        mask = magnitude > 1e-6

        if np.any(mask):
            self.axs[1, 0].quiver(X[mask], Y[mask],
                                 state['velocity_x'][mask],
                                 state['velocity_y'][mask])  # y轴向上为正

        # 设置固定的坐标轴范围和方向
        self.axs[1, 0].set_xlim(0, self.settings.grid_size_x - 1)
        self.axs[1, 0].set_ylim(0, self.settings.grid_size_y - 1)  # y轴范围保持一致

        # 更新侵蚀沉积图
        self.im_erosion.set_array(state['erosion_deposition'])
        self.im_erosion.set_clim(vmin=-0.1, vmax=0.1)
        self.erosion_cbar.update_normal(self.im_erosion)

        self.fig.suptitle(f'水力侵蚀模拟 - 步骤 {step}', fontproperties=font)
        plt.pause(0.01)  # 给matplotlib时间更新图形

    def pause_play(self, event):
        """暂停/播放切换"""
        self.paused = not self.paused
        self.button.label.set_text('Play' if self.paused else 'Pause')

    def show(self):
        """显示图形"""
        plt.show() 