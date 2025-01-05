import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def read_dem_tif(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            # 读取地形数据
            terrain = dataset.read(1)
            
            # 获取地理变换信息
            transform = dataset.transform
            
            # 计算经纬度
            rows, cols = terrain.shape
            lon, lat = np.meshgrid(
                np.arange(cols) * transform[0] + transform[2],
                np.arange(rows) * transform[4] + transform[5]
            )
            
            print(f"成功读取DEM数据：形状 {terrain.shape}")
            print(f"高程范围：{np.min(terrain):.2f}m - {np.max(terrain):.2f}m")
            
            return terrain, lon, lat
    except Exception as e:
        print(f"读取DEM文件失败: {str(e)}")
        return None, None, None

def save_terrain_data(terrain, output_file):
    """
    将地形数据保存为JSON格式
    """
    try:
        # 将numpy数组转换为Python列表
        terrain_data = {
            'width': terrain.shape[1],
            'height': terrain.shape[0],
            'min_height': float(np.min(terrain)),
            'max_height': float(np.max(terrain)),
            'mean_height': float(np.mean(terrain)),
            'heights': terrain.tolist()  # 转换为Python列表
        }
        
        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(terrain_data, f)
            
        print(f"地形数据已保存到: {output_file}")
        print(f"数据统计:")
        print(f"尺寸: {terrain.shape[1]}x{terrain.shape[0]}")
        print(f"最小高程: {terrain_data['min_height']:.2f}m")
        print(f"最大高程: {terrain_data['max_height']:.2f}m")
        print(f"平均高程: {terrain_data['mean_height']:.2f}m")
        
        return True
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建DEM文件的完整路径
    terrain_file = os.path.join(current_dir, 'DEM_50_tianwa.tif')
    
    print(f"正在读取文件: {terrain_file}")
    terrain, lon, lat = read_dem_tif(terrain_file)
    
    if terrain is not None:
        # 删除边缘数据
        terrain = terrain[5:-5, 5:-5]
        lon = lon[5:-5, 5:-5]
        lat = lat[5:-5, 5:-5]
        
        # 保存为JSON文件
        output_file = os.path.join(current_dir, 'terrain_data.json')
        save_terrain_data(terrain, output_file)
        
        # 显示地形图
        plt.figure(figsize=(10, 8))
        plt.imshow(terrain, cmap='terrain')
        plt.colorbar(label='高程 (m)')
        plt.title('DEM地形图')
        plt.xlabel('X (像素)')
        plt.ylabel('Y (像素)')
        plt.show()