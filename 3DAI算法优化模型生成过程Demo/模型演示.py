import torch
import torch.nn as nn

class ImageToPointCloud(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 1024*3)  # 输出1024个点的3D坐标
        )
    def forward(self, x):
        return self.encoder(x).view(-1, 1024, 3)  # 形状：(batch_size, 1024, 3)
    
import open3d as o3d
from PIL import Image
import numpy as np

# 加载图像并预处理
img = Image.open("input.jpg").resize((32, 32))
img = np.array(img) / 255.0
img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)  # 转为PyTorch张量

# 模型推理
model = ImageToPointCloud()
point_cloud = model(img).detach().numpy()[0]  # 获取点云数据

# 可视化
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pcd])