import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from noise import snoise2

# 生成地形
def generate_terrain(width, height, scale=3.0, octaves=10, persistence=0.5, lacunarity=2.0, base_height=5.0):
    """
    生成随机地形数据
    :param width: 
    :param height: 
    :param scale: 
    :param octaves: 
    :param persistence: 
    :param lacunarity: 
    :param base_height: 
    :return:
    """
    terrain = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            nx = x / width - 0.5
            ny = y / height - 0.5
            elevation = snoise2(nx * scale, ny * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            terrain[y][x] = (elevation + 1) / 2 * 50 + base_height  # 将高度范围调整

    terrain = gaussian_filter(terrain, sigma=2)  # 添加平滑

    return terrain

# 在terrain范围内，生成二维河水水源区域
def generate_river_area(terrain):
    # river_area 是bool类型，表示是否是河水区域
    river_area = np.zeros_like(terrain, dtype=bool)
    # 在这里修改生成河水区域的逻辑
    # 在地图中间，生成一个半径为5的圆圈，作为河水区域
    center_x = terrain.shape[1] // 2
    center_y = terrain.shape[0] // 2
    radius = 5
    for y in range(terrain.shape[0]):
        for x in range(terrain.shape[1]):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                river_area[y][x] = True
    return river_area


def save_terrain(terrain, filename):
    np.savetxt(filename, terrain, fmt='%.2f')

def save_river_area(river_area, filename):
    np.savetxt(filename, river_area, fmt='%d')

if __name__ == "__main__":
    width, height = 200, 150
    terrain = generate_terrain(width, height)
    save_terrain(terrain, "terrain_data.txt")

    river_area = generate_river_area(terrain)
    save_river_area(river_area, "river_area.txt")

    # 在一个fig渲染地形和河水区域
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    # 渲染地形
    terrain_img = ax[0].imshow(terrain, cmap='terrain', origin='lower')
    ax[0].set_title("Terrain Height Field")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    fig.colorbar(terrain_img, ax=ax[0], label='Height')  # 使用 fig.colorbar 而不是 ax.colorbar

    # 渲染河水区域
    river_img = ax[1].imshow(river_area, cmap='gray', origin='lower')
    ax[1].set_title("River Area")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()