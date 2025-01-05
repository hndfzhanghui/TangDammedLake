import os
import rasterio
import numpy as np
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS

app = Flask(__name__)

# 详细的 CORS 配置
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/dem/upload', methods=['POST', 'OPTIONS'])
def upload_dem():
    # 处理 OPTIONS 请求
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        if not file.filename.lower().endswith(('.tif', '.tiff')):
            return jsonify({'error': '只支持 TIF/TIFF 格式文件'}), 400
        
        # 保存上传的文件
        filename = os.path.basename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 读取 TIF 文件
        with rasterio.open(filepath) as dataset:
            # 读取高程数据
            elevation_data = dataset.read(1)  # 读取第一个波段
            
            # 获取数据的基本信息
            height, width = elevation_data.shape
            transform = dataset.transform
            
            # 计算实际坐标
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            # 转换为列表格式
            data = {
                'width': width,
                'height': height,
                'cellSize': abs(transform[0]),  # 网格大小
                'elevation': elevation_data.tolist(),
                'bounds': {
                    'left': min(xs[0]),
                    'right': max(xs[0]),
                    'top': max(ys[:,0]),
                    'bottom': min(ys[:,0])
                }
            }
            
            # 清理上传的文件
            os.remove(filepath)
            
            # 添加 CORS 头部到响应
            response = jsonify(data)
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
            return response
            
    except Exception as e:
        # 确保清理上传的文件
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
