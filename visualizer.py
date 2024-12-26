from abc import ABC, abstractmethod

class Visualizer(ABC):
    """可视化器基类"""
    
    @abstractmethod
    def initialize(self, settings, initial_state):
        """初始化可视化器"""
        pass
    
    @abstractmethod
    def update(self, step, state):
        """更新可视化"""
        pass
    
    @abstractmethod
    def show(self):
        """显示可视化窗口"""
        pass 