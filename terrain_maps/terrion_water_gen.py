import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 生成地形高度的函数
def generate_terrain(width, height, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, base_height=0.0):
    """
    生成二维地形高度数组

    Args:
        width: 地形宽度
        height: 地形高度
        scale: 噪声的缩放因子
        octaves: 噪声的层数
        persistence: 噪声的持续度
        lacunarity: 噪声的空间频率
        base_height: 基础高度

    Returns:
        二维地形高度数组
    """
    terrain = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            nx = x / width - 0.5
            ny = y / height - 0.5
            # 使用 Simplex 噪声生成地形高度
            elevation = snoise2(nx * scale, ny * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            # 将噪声值映射到 [0, 1] 范围，并乘以 50，再加上基础高度
            terrain[y][x] = (elevation + 1) / 2 * 50 + base_height

    # 使用高斯滤波器进行平滑处理
    terrain = gaussian_filter(terrain, sigma=2)

    return terrain

# 生成河水区域的函数
def generate_river_area(terrain):
    """
    生成二维河水水源区域

    Args:
        terrain: 二维地形高度数组

    Returns:
        二维布尔数组，表示是否是河水区域
    """
    # 初始化一个与地形高度数组形状相同的布尔数组
    river_area = np.zeros_like(terrain, dtype=bool)
    # 在地图中间，生成一个半径为5的圆圈，作为河水区域
    center_x = terrain.shape[1] // 2
    center_y = terrain.shape[0] // 2
    radius = 5
    for y in range(terrain.shape[0]):
        for x in range(terrain.shape[1]):
            # 判断当前点是否在圆圈内
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                river_area[y][x] = True
    return river_area

# 保存地形高度数组到文件
def save_terrain(terrain, filename):
    np.savetxt(filename, terrain, fmt='%.2f')

# 保存河水区域数组到文件
def save_river_area(river_area, filename):
    np.savetxt(filename, river_area, fmt='%d')

if __name__ == "__main__":
    # 设置地形的宽度和高度
    width, height = 200, 150
    # 生成地形高度数组
    terrain = generate_terrain(width, height)
    # 保存地形高度数组到文件
    save_terrain(terrain, "terrain_data1.txt")

    # 生成河水区域数组
    river_area = generate_river_area(terrain)
    # 保存河水区域数组到文件
    save_river_area(river_area, "river_area1.txt")

    # 创建一个包含两个子图的图形
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    # 在第一个子图中绘制地形高度
    terrain_img = ax[0].imshow(terrain, cmap='terrain', origin='lower')
    ax[0].set_title("Terrain Height Field")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    fig.colorbar(terrain_img, ax=ax[0], label='Height')

    # 在第二个子图中绘制河水区域
    river_img = ax[1].imshow(river_area, cmap='gray', origin='lower')
    ax[1].set_title("River Area")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()