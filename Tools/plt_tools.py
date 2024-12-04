import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
"""
功能:显示多个 cv2 格式的图像在一个图形中。
参数:
    - images: 图像数据的列表，每个图像都是 numpy.ndarray 类型。
    - cols: 每行显示的图像数量，默认为3。
    - figsize: 图形的大小，默认为(15, 10)。
    - titles: 图像标题的列表，如果提供，长度应与图像数量相同。
"""
def PLT_Imgs(images, lines=3, figsize=(15, 10), titles=None):
    # 计算行数
    if lines <=2:
        lines = 2
    num_images = len(images)
    rows = int(num_images / lines) + (num_images % lines > 0)
    # 创建图形和子图
    fig, axs = plt.subplots(rows, lines, figsize=figsize)
    # 如果只有一个子图，将其转换为列表
    if isinstance(axs, plt.Axes):
        axs = [axs]
    # 展平子图列表
    axs = axs.flatten()
    # 显示图像
    for i, img in enumerate(images):
        # 将 BGR 图像转换为 RGB 图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img_rgb)
        if titles and i < len(titles):
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    # 隐藏多余的子图
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])
    # 显示图形
    plt.show()
if __name__ == '__main__':
    img = np.random.randint(0, 255, size=(30,30,3), dtype=np.uint8)
    PLT_Imgs([img,img,img,img],lines=5)